# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import (
    ReplicaId,
    ShardedStateDict,
    ShardedTensorFactory,
)
from megatron.core.fusions.fused_bias_geglu import bias_geglu_impl
from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass
class MLPSubmodules:
    linear_fc1: Union[ModuleSpec, type] = None
    linear_fc2: Union[ModuleSpec, type] = None


class MLP(MegatronModule):
    """
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.


    Returns an output and a bias to be added to the output.
    If config.add_bias_linear is False, the bias returned is None.

    We use the following notation:
     h: hidden size
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        is_expert: bool = False,
        input_size: int = None,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config

        self.input_size = input_size if input_size != None else self.config.hidden_size

        # If this is a gated linear unit we double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        ffn_hidden_size = self.config.ffn_hidden_size
        if self.config.gated_linear_unit:
            ffn_hidden_size *= 2

        self.linear_fc1 = build_module(
            submodules.linear_fc1,
            self.input_size,
            ffn_hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name="fc1",
        )

        self.activation_func = self.config.activation_func

        self.linear_fc2 = build_module(
            submodules.linear_fc2,
            self.config.ffn_hidden_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name="fc2",
        )

    def forward(self, hidden_states):

        # [s, b, 4 * h/p]
        intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)

        if self.config.bias_activation_fusion:
            if self.activation_func == F.gelu:
                if self.config.gated_linear_unit:
                    intermediate_parallel = bias_geglu_impl(intermediate_parallel, bias_parallel)
                else:
                    assert self.config.add_bias_linear is True
                    intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
            elif self.activation_func == F.silu and self.config.gated_linear_unit:
                intermediate_parallel = bias_swiglu_impl(
                    intermediate_parallel,
                    bias_parallel,
                    self.config.activation_func_fp8_input_store,
                )
            else:
                raise ValueError("Only support fusion of gelu and swiglu")
        else:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            if self.config.gated_linear_unit:

                def glu(x):
                    x = torch.chunk(x, 2, dim=-1)
                    return self.config.activation_func(x[0]) * x[1]

                intermediate_parallel = glu(intermediate_parallel)
            else:
                intermediate_parallel = self.activation_func(intermediate_parallel)

        # [s, b, h]
        output, output_bias = self.linear_fc2(intermediate_parallel)

        return output, output_bias

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        sharded_state_dict = {}
        for name, module in self._modules.items():
            if name == "linear_fc1" and self.config.gated_linear_unit:
                sub_sd = self._sharded_state_dict_for_glu(name, module, prefix, sharded_offsets, metadata)
            else:
                sub_sd = module.sharded_state_dict(f"{prefix}{name}.", sharded_offsets, metadata)
            sharded_state_dict.update(sub_sd)
        return sharded_state_dict

    def _sharded_state_dict_for_glu(
        self,
        module_name: str,
        module: torch.nn.Module,
        prefix: str,
        sharded_offsets: Tuple[Tuple[int, int, int]],
        metadata: Optional[dict] = None,
    ):
        assert module_name == "linear_fc1", module_name
        sharded_state_dict = module.sharded_state_dict(f"{prefix}{module_name}.", sharded_offsets, metadata)
        weight_key = f"{prefix}{module_name}.weight"
        prev_sh_ten = sharded_state_dict[weight_key]

        # We must split the tensor into 2 parts, each sharded separately.
        # This requires a ShardedTensorFactory which `chunk`s during saving
        # and `cat`s during loading
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        tp_size = parallel_state.get_tensor_model_parallel_world_size()

        tp_shard_axis = 0
        prepend_axis_num = len(sharded_offsets)

        def sh_ten_build_fn(key: str, t: torch.Tensor, replica_id: ReplicaId):
            offset_w = (tp_shard_axis + prepend_axis_num, tp_rank, tp_size * 2)
            offset_v = (tp_shard_axis + prepend_axis_num, tp_size + tp_rank, tp_size * 2)
            with torch.no_grad():
                tensor_w, tensor_v = torch.chunk(t, 2, dim=tp_shard_axis)
            return [
                ShardedTensor.from_rank_offsets(
                    key,
                    tensor_w,
                    *sharded_offsets,
                    offset_w,
                    replica_id=replica_id,
                    prepend_axis_num=prepend_axis_num,
                ),
                ShardedTensor.from_rank_offsets(
                    key,
                    tensor_v,
                    *sharded_offsets,
                    offset_v,
                    replica_id=replica_id,
                    prepend_axis_num=prepend_axis_num,
                ),
            ]

        def sh_ten_merge_fn(sub_state_dict):
            with torch.no_grad():
                return torch.cat(sub_state_dict)

        sharded_state_dict[weight_key] = ShardedTensorFactory(
            prev_sh_ten.key,
            prev_sh_ten.data,
            sh_ten_build_fn,
            sh_ten_merge_fn,
            prev_sh_ten.replica_id,
        )
        return sharded_state_dict
