#!/usr/bin/env python3
"""
Standalone Megatron to HuggingFace conversion for MiniCPM4
This implementation doesn't require the full Megatron-LM environment
"""

import os
import torch
import json
from pathlib import Path
from typing import Dict, List, Optional


def convert_minicpm4_megatron_to_hf_standalone(
    checkpoint_path: str,
    output_path: str,
    num_layer: int,
    tp_size: int,
    pp_size: int,
    num_kv_heads: int,
    num_query_heads: int,
    hidden_size: int = 4096,
    vocab_size: int = 73448,
    intermediate_size: int = 16384,
    head_dim: int = 64,
    **kwargs,
) -> None:
    """
    Standalone conversion of MiniCPM-4 Megatron checkpoint to HuggingFace format
    
    Args:
        checkpoint_path: Path to the Megatron checkpoint directory
        output_path: Path to save the HF weights
        num_layer: Number of transformer layers
        tp_size: Tensor parallel size
        pp_size: Pipeline parallel size
        num_kv_heads: Number of KV attention heads
        num_query_heads: Number of query attention heads
        hidden_size: Hidden size of the model
        vocab_size: Vocabulary size
        intermediate_size: Intermediate size for MLP
        head_dim: Dimension of each attention head
        **kwargs: Additional arguments
    """
    print(f"Converting MiniCPM-4 Megatron to HF (Standalone): {checkpoint_path} -> {output_path}")
    print(f"Model config: {num_layer} layers, TP={tp_size}, PP={pp_size}")
    print(f"Attention config: {num_query_heads} query heads, {num_kv_heads} KV heads")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load and merge all tensor parallel shards
    merged_state_dict = {}
    
    for tp_rank in range(tp_size):
        for pp_rank in range(pp_size):
            if pp_size != 1:
                rank_dir = f"mp_rank_0{tp_rank}_00{pp_rank}"
            else:
                rank_dir = f"mp_rank_0{tp_rank}"
            
            checkpoint_file = Path(checkpoint_path) / rank_dir / "model_optim_rng.pt"
            
            if checkpoint_file.exists():
                print(f"Loading checkpoint from {checkpoint_file}")
                checkpoint = torch.load(checkpoint_file, map_location="cpu")
                state_dict = checkpoint["model"]
                
                # Merge tensor parallel shards
                for key, value in state_dict.items():
                    if key not in merged_state_dict:
                        merged_state_dict[key] = []
                    merged_state_dict[key].append(value)
    
    # Concatenate tensor parallel shards
    hf_state_dict = {}
    
    # Handle embedding
    if "embedding.word_embeddings.weight" in merged_state_dict:
        embedding_shards = merged_state_dict["embedding.word_embeddings.weight"]
        hf_state_dict["model.embed_tokens.weight"] = torch.cat(embedding_shards, dim=0)
        print(f"✓ Merged embedding: {hf_state_dict['model.embed_tokens.weight'].shape}")
    
    # Handle final layer norm
    if "decoder.final_layernorm.weight" in merged_state_dict:
        hf_state_dict["model.norm.weight"] = merged_state_dict["decoder.final_layernorm.weight"][0]
        print(f"✓ Final layer norm: {hf_state_dict['model.norm.weight'].shape}")
    
    # Handle layers
    num_query_heads_per_group = num_query_heads // num_kv_heads
    
    for layer_idx in range(num_layer):
        print(f"Processing layer {layer_idx}...")
        
        # Input layer norm
        if f"decoder.layers.{layer_idx}.input_layernorm.weight" in merged_state_dict:
            hf_state_dict[f"model.layers.{layer_idx}.input_layernorm.weight"] = \
                merged_state_dict[f"decoder.layers.{layer_idx}.input_layernorm.weight"][0]
        
        # Attention layers
        if f"decoder.layers.{layer_idx}.self_attention.linear_qkv.weight" in merged_state_dict:
            qkv_weight = merged_state_dict[f"decoder.layers.{layer_idx}.self_attention.linear_qkv.weight"][0]
            
            # Split QKV weight back to separate projections
            # The format is: [q1, q2, ..., qN, k1, k2, ..., kN, v1, v2, ..., vN]
            total_heads = num_query_heads + num_kv_heads + num_kv_heads  # q + k + v
            head_dim = qkv_weight.shape[0] // total_heads
            
            qkv_split = torch.split(qkv_weight, split_size_or_sections=head_dim, dim=0)
            
            # Extract Q, K, V projections
            q_start = 0
            q_end = num_query_heads
            k_start = num_query_heads
            k_end = num_query_heads + num_kv_heads
            v_start = num_query_heads + num_kv_heads
            v_end = num_query_heads + num_kv_heads + num_kv_heads
            
            q_proj_list = qkv_split[q_start:q_end]
            k_proj_list = qkv_split[k_start:k_end]
            v_proj_list = qkv_split[v_start:v_end]
            
            hf_state_dict[f"model.layers.{layer_idx}.self_attn.q_proj.weight"] = torch.cat(q_proj_list, dim=0)
            hf_state_dict[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = torch.cat(k_proj_list, dim=0)
            hf_state_dict[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = torch.cat(v_proj_list, dim=0)
        
        # Output projection
        if f"decoder.layers.{layer_idx}.self_attention.linear_proj.weight" in merged_state_dict:
            o_proj_shards = merged_state_dict[f"decoder.layers.{layer_idx}.self_attention.linear_proj.weight"]
            hf_state_dict[f"model.layers.{layer_idx}.self_attn.o_proj.weight"] = torch.cat(o_proj_shards, dim=1)
        
        # Post attention layer norm
        if f"decoder.layers.{layer_idx}.pre_mlp_layernorm.weight" in merged_state_dict:
            hf_state_dict[f"model.layers.{layer_idx}.post_attention_layernorm.weight"] = \
                merged_state_dict[f"decoder.layers.{layer_idx}.pre_mlp_layernorm.weight"][0]
        
        # MLP layers (dense layers)
        if f"decoder.layers.{layer_idx}.mlp.linear_fc1.weight" in merged_state_dict:
            fc1_weight = merged_state_dict[f"decoder.layers.{layer_idx}.mlp.linear_fc1.weight"][0]
            fc2_shards = merged_state_dict[f"decoder.layers.{layer_idx}.mlp.linear_fc2.weight"]
            
            # Split fc1 for SwiGLU
            gate_proj, up_proj = torch.split(fc1_weight, split_size_or_sections=fc1_weight.shape[0] // 2, dim=0)
            
            hf_state_dict[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = gate_proj
            hf_state_dict[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = up_proj
            hf_state_dict[f"model.layers.{layer_idx}.mlp.down_proj.weight"] = torch.cat(fc2_shards, dim=1)
    
    # Save HuggingFace format
    torch.save(hf_state_dict, os.path.join(output_path, "pytorch_model.bin"))
    print(f"✓ Saved HF weights to {os.path.join(output_path, 'pytorch_model.bin')}")
    
    # Create config file
    config = {
        "architectures": ["MiniCPMForCausalLM"],
        "model_type": "minicpm",
        "hidden_size": hidden_size,
        "num_attention_heads": num_query_heads,
        "num_hidden_layers": num_layer,
        "intermediate_size": intermediate_size,
        "max_position_embeddings": 32768,
        "vocab_size": vocab_size,
        "rms_norm_eps": 1e-5,
        "use_cache": True,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.46.3",
    }
    
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"✓ Saved config to {os.path.join(output_path, 'config.json')}")
    
    # Create tokenizer config (basic)
    tokenizer_config = {
        "model_type": "minicpm",
        "tokenizer_class": "LlamaTokenizer",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "bos_token": "<s>",
        "eos_token": "</s>",
    }
    
    with open(os.path.join(output_path, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f, indent=2)
    print(f"✓ Saved tokenizer config to {os.path.join(output_path, 'tokenizer_config.json')}")
    
    print(f"✓ Conversion completed! Total parameters: {sum(p.numel() for p in hf_state_dict.values())}")
    print(f"✓ Output directory: {output_path}")


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
    Main conversion function - tries distributed version first, falls back to standalone
    """
    try:
        # Try to use the distributed version if available
        from .dist_ckpt_to_hf_minicpm4 import convert_minicpm4_megatron_to_hf as dist_convert
        print("Using distributed Megatron->HF conversion...")
        dist_convert(
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            num_layer=num_layer,
            tp_size=tp_size,
            pp_size=pp_size,
            num_kv_heads=num_kv_heads,
            num_query_heads=num_query_heads,
            **kwargs
        )
    except ImportError:
        # Fall back to standalone version
        print("Using standalone Megatron->HF conversion...")
        convert_minicpm4_megatron_to_hf_standalone(
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            num_layer=num_layer,
            tp_size=tp_size,
            pp_size=pp_size,
            num_kv_heads=num_kv_heads,
            num_query_heads=num_query_heads,
            **kwargs
        )
    except Exception as e:
        print(f"Error in distributed conversion: {e}")
        print("Falling back to standalone conversion...")
        convert_minicpm4_megatron_to_hf_standalone(
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            num_layer=num_layer,
            tp_size=tp_size,
            pp_size=pp_size,
            num_kv_heads=num_kv_heads,
            num_query_heads=num_query_heads,
            **kwargs
        ) 