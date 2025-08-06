#!/usr/bin/env python3
"""
MiniCPM Checkpoint to HuggingFace Converter with Tensor Parallel and Pipeline Parallel Support
Converts distributed Megatron checkpoints to HuggingFace format
"""

import argparse
import os
import torch
from typing import List, Dict


def load_distributed_checkpoints(
    in_dir: str, tp_size: int, pp_size: int, num_layer: int
) -> List[Dict[str, torch.Tensor]]:
    """
    Load distributed checkpoints from TP/PP sharded files

    Args:
        in_dir: Input directory containing checkpoint files
        tp_size: Tensor parallel size
        pp_size: Pipeline parallel size
        num_layer: Total number of transformer layers

    Returns:
        List of state dictionaries for each TP rank
    """
    print(f"Loading distributed checkpoints: TP={tp_size}, PP={pp_size}")

    # Initialize TP world state dictionaries
    ckpt_tp_world = [{} for _ in range(tp_size)]

    # Calculate layers per pipeline stage
    num_layer_per_pp = num_layer // pp_size

    # Load sharded weights
    for pp_rank in range(pp_size):
        for tp_rank in range(tp_size):
            # Determine checkpoint path based on parallel configuration
            if pp_size == 1:
                ckpt_path = os.path.join(in_dir, f"mp_rank_{tp_rank:02d}/model_optim_rng.pt")
            else:
                ckpt_path = os.path.join(in_dir, f"mp_rank_{tp_rank:02d}_{pp_rank:03d}/model_optim_rng.pt")

            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

            print(f"Loading: {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location="cpu")["model"]

            # Process layer indices for pipeline parallelism
            layer_idx_base = pp_rank * num_layer_per_pp
            new_state_dict = {}

            for key in state_dict:
                if "extra" not in key:
                    print(f"Processing key: {key}, shape: {state_dict[key].shape}")
                    if key.startswith("decoder.layers."):
                        layer_idx_abs = int(key.split(".")[2])
                        layer_idx = layer_idx_base + layer_idx_abs
                        new_key = (
                            ".".join(key.split(".")[:2]) + "." + str(layer_idx) + "." + ".".join(key.split(".")[3:])
                        )
                        new_state_dict[new_key] = state_dict[key]
                    else:
                        new_state_dict[key] = state_dict[key]

            # Store in TP world
            for key in new_state_dict:
                assert key not in ckpt_tp_world[tp_rank], f"Duplicate key found: {key}"
                ckpt_tp_world[tp_rank][key] = new_state_dict[key]

    return ckpt_tp_world


def convert_minicpm_megatron_to_hf_tp_pp(
    num_layer: int, tp_size: int, pp_size: int, in_dir: str, save_path: str, num_kv_heads: int, num_query_heads: int
) -> None:
    """
    Convert MiniCPM Megatron checkpoint to HuggingFace format with TP/PP support

    Args:
        num_layer: Number of transformer layers
        tp_size: Tensor parallel size
        pp_size: Pipeline parallel size
        in_dir: Input Megatron checkpoint directory
        save_path: Output HF weights file path
        num_kv_heads: Number of KV attention heads
        num_query_heads: Number of query attention heads
    """
    print(f"Starting conversion: {in_dir} -> {save_path}")
    print(f"Model config: {num_layer} layers, TP={tp_size}, PP={pp_size}")
    print(f"Attention config: {num_query_heads} query heads, {num_kv_heads} KV heads")

    # Validate parameters
    assert num_query_heads % num_kv_heads == 0, "Query heads must be divisible by KV heads"
    assert num_layer % pp_size == 0, "Number of layers must be divisible by PP size"
    assert num_query_heads % tp_size == 0, "Query heads must be divisible by TP size"
    assert num_kv_heads % tp_size == 0, "KV heads must be divisible by TP size"

    # Load distributed checkpoints
    ckpt_tp_world = load_distributed_checkpoints(in_dir, tp_size, pp_size, num_layer)

    # Initialize HF model state dict
    hf_model_state_dict = {}

    # Merge embedding and output layer weights
    embedding_list, lm_head_list = [], []
    for sd in ckpt_tp_world:
        embedding_list.append(sd["embedding.word_embeddings.weight"])
        lm_head_list.append(sd["output_layer.weight"])

    hf_model_state_dict["model.embed_tokens.weight"] = torch.cat(embedding_list, dim=0).cpu()
    hf_model_state_dict["lm_head.weight"] = torch.cat(lm_head_list, dim=0).cpu()
    hf_model_state_dict["model.norm.weight"] = ckpt_tp_world[0]["decoder.final_layernorm.weight"].cpu()

    # Calculate attention parameters
    num_query_heads_per_group = num_query_heads // num_kv_heads
    num_group_per_tp_rank = num_kv_heads // tp_size
    dim_model = hf_model_state_dict["model.embed_tokens.weight"].shape[1]

    # Convert each layer
    for layer_idx in range(num_layer):
        print(f"Processing layer {layer_idx}")

        # Input layer normalization
        hf_model_state_dict[f"model.layers.{layer_idx}.input_layernorm.weight"] = ckpt_tp_world[0][
            f"decoder.layers.{layer_idx}.self_attention.linear_qkv.layer_norm_weight"
        ].cpu()

        # Attention layer weights
        q_proj_list, k_proj_list, v_proj_list, o_proj_list = [], [], [], []
        q_bias_list, k_bias_list, v_bias_list = [], [], []

        for sd in ckpt_tp_world:
            qkv_proj = sd[f"decoder.layers.{layer_idx}.self_attention.linear_qkv.weight"]
            qkv_bias = sd[f"decoder.layers.{layer_idx}.self_attention.linear_qkv.bias"]

            # Split by head dimension
            head_dim = dim_model // num_query_heads
            qkv_proj_split = torch.split(qkv_proj, split_size_or_sections=head_dim, dim=0)
            qkv_bias_split = torch.split(qkv_bias, split_size_or_sections=head_dim, dim=0)

            # Reorganize QKV weights
            for i in range(num_group_per_tp_rank):
                k_proj_list.append(qkv_proj_split[(num_query_heads_per_group + 2) * i + num_query_heads_per_group])
                v_proj_list.append(qkv_proj_split[(num_query_heads_per_group + 2) * i + num_query_heads_per_group + 1])
                q_proj_list.extend(
                    qkv_proj_split[
                        (num_query_heads_per_group + 2) * i : (num_query_heads_per_group + 2) * i
                        + num_query_heads_per_group
                    ]
                )

                k_bias_list.append(qkv_bias_split[(num_query_heads_per_group + 2) * i + num_query_heads_per_group])
                v_bias_list.append(qkv_bias_split[(num_query_heads_per_group + 2) * i + num_query_heads_per_group + 1])
                q_bias_list.extend(
                    qkv_bias_split[
                        (num_query_heads_per_group + 2) * i : (num_query_heads_per_group + 2) * i
                        + num_query_heads_per_group
                    ]
                )

            o_proj_list.append(sd[f"decoder.layers.{layer_idx}.self_attention.linear_proj.weight"])

        # Merge attention weights
        hf_model_state_dict[f"model.layers.{layer_idx}.self_attn.q_proj.weight"] = torch.cat(q_proj_list, dim=0).cpu()
        hf_model_state_dict[f"model.layers.{layer_idx}.self_attn.q_proj.bias"] = torch.cat(q_bias_list, dim=0).cpu()
        hf_model_state_dict[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = torch.cat(k_proj_list, dim=0).cpu()
        hf_model_state_dict[f"model.layers.{layer_idx}.self_attn.k_proj.bias"] = torch.cat(k_bias_list, dim=0).cpu()
        hf_model_state_dict[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = torch.cat(v_proj_list, dim=0).cpu()
        hf_model_state_dict[f"model.layers.{layer_idx}.self_attn.v_proj.bias"] = torch.cat(v_bias_list, dim=0).cpu()
        hf_model_state_dict[f"model.layers.{layer_idx}.self_attn.o_proj.weight"] = torch.cat(o_proj_list, dim=1).cpu()

        # MLP layer weights
        hf_model_state_dict[f"model.layers.{layer_idx}.post_attention_layernorm.weight"] = ckpt_tp_world[0][
            f"decoder.layers.{layer_idx}.mlp.linear_fc1.layer_norm_weight"
        ].cpu()

        gate_proj_list, up_proj_list, down_proj_list = [], [], []
        for sd in ckpt_tp_world:
            linear1_fc_weight = sd[f"decoder.layers.{layer_idx}.mlp.linear_fc1.weight"]
            gate_proj, up_proj = torch.split(
                linear1_fc_weight, split_size_or_sections=(linear1_fc_weight.shape[0] // 2), dim=0
            )
            gate_proj_list.append(gate_proj)
            up_proj_list.append(up_proj)
            down_proj_list.append(sd[f"decoder.layers.{layer_idx}.mlp.linear_fc2.weight"])

        hf_model_state_dict[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = torch.cat(gate_proj_list, dim=0).cpu()
        hf_model_state_dict[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = torch.cat(up_proj_list, dim=0).cpu()
        hf_model_state_dict[f"model.layers.{layer_idx}.mlp.down_proj.weight"] = torch.cat(down_proj_list, dim=1).cpu()

    # Print final weight shapes
    print("Final weight shapes:")
    for key in hf_model_state_dict:
        print(f"{key}: {hf_model_state_dict[key].shape}")

    # Save converted weights
    print(f"Saving converted weights to: {save_path}")
    torch.save(hf_model_state_dict, save_path)

    # Print statistics
    total_params = sum(p.numel() for p in hf_model_state_dict.values())
    print(f"Conversion completed! Total parameters: {total_params:,}")
    print(f"Weight file size: {os.path.getsize(save_path) / 1024 / 1024 / 1024:.2f} GB")


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="MiniCPM Megatron to HuggingFace conversion with TP/PP support")
    parser.add_argument("--num_layer", type=int, default=80, help="Number of transformer layers")
    parser.add_argument("--tp_size", type=int, default=8, help="Tensor parallel size")
    parser.add_argument("--pp_size", type=int, default=4, help="Pipeline parallel size")
    parser.add_argument("--in_dir", type=str, default="", help="Input Megatron checkpoint directory")
    parser.add_argument("--save_path", type=str, default="", help="Output HF weights file path")
    parser.add_argument("--num_kv_heads", type=int, default=8, help="Number of KV attention heads")
    parser.add_argument("--num_query_heads", type=int, default=64, help="Number of query attention heads")

    args = parser.parse_args()

    if not args.in_dir or not args.save_path:
        print("Error: --in_dir and --save_path are required")
        return

    convert_minicpm_megatron_to_hf_tp_pp(
        num_layer=args.num_layer,
        tp_size=args.tp_size,
        pp_size=args.pp_size,
        in_dir=args.in_dir,
        save_path=args.save_path,
        num_kv_heads=args.num_kv_heads,
        num_query_heads=args.num_query_heads,
    )


if __name__ == "__main__":
    main()
