"""Pretrain GPT."""

import os
import sys

sys.path.append("./")
import torch
import copy
import megatron
import yaml

from types import SimpleNamespace
from typing import Union
from megatron.core import mpu
from megatron.training import get_args
from megatron.core.enums import ModelType
from megatron.training import print_rank_0
from megatron.core.models.gpt import GPTModel
from megatron.core.models.minillm import DistilModel
from megatron.core.utils import StragglerDetector
from megatron.training.initialize import initialize_megatron
from megatron.core.transformer.spec_utils import import_module
from megatron.training.training import setup_model_and_optimizer
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_layer_with_mla_spec,
)


stimer = StragglerDetector()


def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel, DistilModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()

    print_rank_0("building Distil model ...")

    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        transformer_config = core_transformer_config_from_yaml(args, "language_model")
    else:
        transformer_config = core_transformer_config_from_args(args)

    if args.spec is not None:
        transformer_layer_spec = import_module(args.spec)
    else:
        if args.use_mla:
            transformer_layer_spec = get_gpt_layer_with_mla_spec(
                args.num_experts, args.moe_grouped_gemm, use_block_ffn=args.use_block_ffn
            )
        else:
            if args.transformer_impl == "transformer_engine":
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                    args.num_experts, args.moe_grouped_gemm, args.qk_layernorm, use_block_ffn=args.use_block_ffn
                )
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(
                    args.num_experts, args.moe_grouped_gemm, args.qk_layernorm, use_block_ffn=args.use_block_ffn
                )

    if args.distillation:
        assert args.teacher_yaml_cfg is not None, "Please check the teacher config"
        teacher_transformer_config = copy.deepcopy(transformer_config)

        with open(args.teacher_yaml_cfg, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            teacher_transformer_config = vars(teacher_transformer_config)
            teacher_transformer_config.update(config)
            teacher_transformer_config = SimpleNamespace(**teacher_transformer_config)

        if teacher_transformer_config.use_mla:
            teacher_transformer_layer_spec = get_gpt_layer_with_mla_spec(
                teacher_transformer_config.num_experts,
                teacher_transformer_config.moe_grouped_gemm,
                use_block_ffn=teacher_transformer_config.use_block_ffn,
            )
        else:
            if teacher_transformer_config.transformer_impl == "transformer_engine":
                teacher_transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                    teacher_transformer_config.num_experts,
                    teacher_transformer_config.moe_grouped_gemm,
                    teacher_transformer_config.qk_layernorm,
                    use_block_ffn=teacher_transformer_config.use_block_ffn,
                )
            else:
                teacher_transformer_layer_spec = get_gpt_layer_local_spec(
                    teacher_transformer_config.num_experts,
                    teacher_transformer_config.moe_grouped_gemm,
                    teacher_transformer_config.qk_layernorm,
                    use_block_ffn=teacher_transformer_config.use_block_ffn,
                )

        print(f"teacher_transformer_config-{teacher_transformer_config}")
        print(f"transformer_config-{transformer_config}")

        model = DistilModel(
            teacher_transformer_config=teacher_transformer_config,
            teacher_transformer_layer_spec=teacher_transformer_layer_spec,
            transformer_config=transformer_config,
            transformer_layer_spec=transformer_layer_spec,
            max_sequence_length=args.max_position_embeddings,
            vocab_size=args.padded_vocab_size if not args.use_modelbest_sdk else args.vocab_size,
            allow_missing_student_model_checkpoint=args.allow_missing_student_model_checkpoint,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            pre_process=pre_process,
            post_process=post_process,
            parallel_output=True,
            distil_loss_type=args.distil_loss_type,
        )
        model.freeze()
    else:
        model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size if not args.use_modelbest_sdk else args.vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
        )

    return model


if __name__ == "__main__":

    initialize_megatron(extra_args_provider=None, args_defaults={"tokenizer_type": "Llama2Tokenizer"})

    args = get_args()
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
        model_provider, model_type=ModelType.encoder_or_decoder
    )
    model_real = model[0]

    # extract megatron state dict
    state_dict_megatron = model_real.state_dict()

    # print("*******")
    # print(state_dict_megatron.keys())
    # print("*******")

    args.dense_layer_ids = set(map(int, args.dense_layer_ids.split(",")))

    new_sd = dict()
    for k in state_dict_megatron:
        if "_extra" in k:
            continue
        if args.distillation and "teacher" in k:
            continue
        if args.distillation and "student" in k:
            new_sd[k.replace("student_model.", "")] = state_dict_megatron[k]
        else:
            new_sd[k] = state_dict_megatron[k]
        print(k, state_dict_megatron[k].shape)

    state_dict_megatron = new_sd

    # torch.save({"model": new_sd}, args.save)
    # param name mapping
    state_dict_hf = dict()
    state_dict_hf["model.embed_tokens.weight"] = state_dict_megatron["embedding.word_embeddings.weight"]
    state_dict_hf["model.norm.weight"] = state_dict_megatron["decoder.final_layernorm.weight"]
    if args.untie_embeddings_and_output_weights:
        state_dict_hf["lm_head.weight"] = state_dict_megatron["output_layer.weight"]

    assert args.num_attention_heads % args.num_query_groups == 0
    if not args.kv_channels:
        assert args.hidden_size % args.num_attention_heads == 0
        args.kv_channels = args.hidden_size // args.num_attention_heads

    num_query_heads_per_group = args.num_attention_heads // args.num_query_groups

    for layer_idx in range(args.num_layers):
        state_dict_hf[f"model.layers.{layer_idx}.input_layernorm.weight"] = state_dict_megatron[
            f"decoder.layers.{layer_idx}.input_layernorm.weight"
        ]

        qkv_proj = state_dict_megatron[f"decoder.layers.{layer_idx}.self_attention.linear_qkv.weight"]
        qkv_proj_split = torch.split(qkv_proj, split_size_or_sections=args.kv_channels, dim=0)
        q_proj_list, k_proj_list, v_proj_list = [], [], []
        for i in range(args.num_query_groups):
            q_proj_list.extend(
                qkv_proj_split[
                    (num_query_heads_per_group + 2) * i : (num_query_heads_per_group + 2) * i
                    + num_query_heads_per_group
                ]
            )
            k_proj_list.append(qkv_proj_split[(num_query_heads_per_group + 2) * i + num_query_heads_per_group])
            v_proj_list.append(qkv_proj_split[(num_query_heads_per_group + 2) * i + num_query_heads_per_group + 1])
        q_proj = torch.cat(q_proj_list, dim=0)
        k_proj = torch.cat(k_proj_list, dim=0)
        v_proj = torch.cat(v_proj_list, dim=0)

        state_dict_hf[f"model.layers.{layer_idx}.self_attn.q_proj.weight"] = q_proj
        state_dict_hf[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = k_proj
        state_dict_hf[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = v_proj
        state_dict_hf[f"model.layers.{layer_idx}.self_attn.o_proj.weight"] = state_dict_megatron[
            f"decoder.layers.{layer_idx}.self_attention.linear_proj.weight"
        ]
        state_dict_hf[f"model.layers.{layer_idx}.post_attention_layernorm.weight"] = state_dict_megatron[
            f"decoder.layers.{layer_idx}.pre_mlp_layernorm.weight"
        ]

        if layer_idx + 1 in args.dense_layer_ids:
            linear1_fc_weight = state_dict_megatron[f"decoder.layers.{layer_idx}.mlp.linear_fc1.weight"]
            gate_proj, up_proj = torch.split(
                linear1_fc_weight, split_size_or_sections=(linear1_fc_weight.shape[0] // 2), dim=0
            )
            state_dict_hf[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = gate_proj
            state_dict_hf[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = up_proj
            state_dict_hf[f"model.layers.{layer_idx}.mlp.down_proj.weight"] = state_dict_megatron[
                f"decoder.layers.{layer_idx}.mlp.linear_fc2.weight"
            ]
        else:
            state_dict_hf[f"model.layers.{layer_idx}.mlp.moe_router.weight"] = state_dict_megatron[
                f"decoder.layers.{layer_idx}.mlp.moe_router.weight"
            ]
            state_dict_hf[f"model.layers.{layer_idx}.mlp.router_norm.weight"] = state_dict_megatron[
                f"decoder.layers.{layer_idx}.mlp.router_norm.weight"
            ]
            state_dict_hf[f"model.layers.{layer_idx}.mlp.moe_w_in.weight"] = state_dict_megatron[
                f"decoder.layers.{layer_idx}.mlp.moe_w_in.weight"
            ]
            state_dict_hf[f"model.layers.{layer_idx}.mlp.moe_w_out.weight"] = state_dict_megatron[
                f"decoder.layers.{layer_idx}.mlp.moe_w_out.weight"
            ]

    # save and generate hf repository
    os.makedirs(args.save, exist_ok=True)
    tp_size = args.tensor_model_parallel_size
    if tp_size == 1:
        torch.save(state_dict_hf, os.path.join(args.save, "pytorch_model.bin"))
    else:
        tp_rank = mpu.get_tensor_model_parallel_rank()
        torch.save(state_dict_hf, os.path.join(args.save, f"pytorch_model.tp{tp_rank}.bin"))


def convert_minicpm4_megatron_to_hf(
    checkpoint_path: str,
    output_path: str,
    num_layer: int,
    tp_size: int,
    pp_size: int,
    num_kv_heads: int,
    num_query_heads: int,
    **kwargs,
) -> None:
    """
    Wrapper function for MiniCPM-4 Megatron to HF conversion using distributed checkpointing

    Args:
        checkpoint_path: Path to the Megatron checkpoint directory
        output_path: Path to save the HF weights
        num_layer: Number of transformer layers
        tp_size: Tensor parallel size
        pp_size: Pipeline parallel size
        num_kv_heads: Number of KV attention heads
        num_query_heads: Number of query attention heads
        **kwargs: Additional arguments for the conversion
    """
    # This function would need to be implemented to call the main conversion logic
    # For now, it's a placeholder that can be extended based on the specific needs
    print(f"Converting MiniCPM-4 Megatron to HF: {checkpoint_path} -> {output_path}")
    print(f"Model config: {num_layer} layers, TP={tp_size}, PP={pp_size}")
    print(f"Attention config: {num_query_heads} query heads, {num_kv_heads} KV heads")

    # The actual conversion logic would go here
    # This would involve setting up the Megatron environment and calling the main conversion
    pass
