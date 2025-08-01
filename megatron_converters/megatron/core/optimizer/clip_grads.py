# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Gradient clipping."""

from typing import List, Optional, Union

# import amp_C
# For conversion scripts, we don't need amp_C
amp_C = None
import torch

# from apex.multi_tensor_apply import multi_tensor_applier
# For conversion scripts, we don't need multi_tensor_applier
multi_tensor_applier = None
from torch import inf

from ..tensor_parallel import param_is_not_tensor_parallel_duplicate
from ..transformer.module import param_is_not_shared


def clip_grad_norm_fp32(
    parameters: Union[List[torch.Tensor], torch.Tensor],
    grads_for_norm: Union[List[torch.Tensor], torch.Tensor],
    max_norm: Union[int, float],
    norm_type: Union[int, float] = 2,
    model_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
) -> float:
    """Clips gradient norm of an iterable of parameters whose gradients
       are in fp32.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Note that
    the gradients are modified in place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized.
        grads_for_norm (Iterable[Tensor]): an iterable of Tensors or a single
            Tensor that will be used for calculating the grad norm.
        max_norm (float or int): max norm of the gradients.
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        model_parallel_group (torch.distributed.ProcessGroup, optional): model-parallel
            group over which grad norm needs to be aggregated.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    if isinstance(grads_for_norm, torch.Tensor):
        grads_for_norm = [grads_for_norm]

    # Grads.
    grads = []
    for param in parameters:
        if param.grad is not None:
            assert param.grad.type() == "torch.cuda.FloatTensor"
            grads.append(param.grad.detach())

    # Norm parameters.
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    total_norm = 0.0

    # Calculate norm.
    if norm_type == inf:
        total_norm = max(grad.abs().max() for grad in grads_for_norm)
        total_norm_cuda = torch.tensor([float(total_norm)], dtype=torch.float, device="cuda")
        # Take max across all model-parallel GPUs.
        torch.distributed.all_reduce(total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=model_parallel_group)
        total_norm = total_norm_cuda[0].item()

    else:
        if norm_type == 2.0:
            dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device="cuda")
            # Use apex's multi-tensor applier for efficiency reasons.
            # Multi-tensor applier takes a function and a list of list
            # and performs the operation on that list all in one kernel.
            if grads_for_norm:
                grad_norm, _ = multi_tensor_applier(
                    amp_C.multi_tensor_l2norm,
                    dummy_overflow_buf,
                    [grads_for_norm],
                    False,  # no per-parameter norm
                )
            else:
                grad_norm = torch.tensor([0], dtype=torch.float, device="cuda")
            # Since we will be summing across data parallel groups,
            # we need the pow(norm-type).
            total_norm = grad_norm**norm_type

        else:
            for grad in grads_for_norm:
                grad_norm = torch.norm(grad, norm_type)
                total_norm += grad_norm**norm_type

        # Sum across all model-parallel GPUs.
        torch.distributed.all_reduce(total_norm, op=torch.distributed.ReduceOp.SUM, group=model_parallel_group)
        total_norm = total_norm.item() ** (1.0 / norm_type)

    # Scale.
    clip_coeff = max_norm / (total_norm + 1.0e-6)
    if clip_coeff < 1.0:
        dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device="cuda")
        multi_tensor_applier(amp_C.multi_tensor_scale, dummy_overflow_buf, [grads, grads], clip_coeff)

    return total_norm


def count_zeros_fp32(
    parameters: Union[List[torch.Tensor], torch.Tensor],
    model_parallel_group: torch.distributed.ProcessGroup,
) -> float:
    """Counts the number of zeros in gradients associated with the passed-in list of
    parameters.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have the number of zeros in its corresponding
            gradient counted.
        model_parallel_group (torch.distributed.ProcessGroup, optional): model-parallel
            group over which grad norm needs to be aggregated.
    """

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    # Filter parameters based on:
    #   - grad should not be none
    #   - parameter should not be shared
    #   - should not be a replica due to tensor model parallelism
    total_num_zeros = torch.tensor([0.0], dtype=torch.float, device="cuda")
    for param in parameters:
        grad_not_none = param.grad is not None
        is_not_shared = param_is_not_shared(param)
        is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
        if grad_not_none and is_not_shared and is_not_tp_duplicate:
            grad = param.grad.detach()
            num_zeros = grad.numel() - torch.count_nonzero(grad)
            total_num_zeros = num_zeros + total_num_zeros

    # Sum across all model-parallel GPUs.
    torch.distributed.all_reduce(total_num_zeros, op=torch.distributed.ReduceOp.SUM, group=model_parallel_group)

    total_num_zeros = total_num_zeros.item()

    return total_num_zeros
