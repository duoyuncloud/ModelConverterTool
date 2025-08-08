#!/usr/bin/env python3
"""
Simple reverse conversion script for MiniCPM Megatron to HuggingFace
"""

import torch
import os
import json
from transformers import AutoConfig, AutoModelForCausalLM

def reverse_convert_minicpm_megatron_to_hf(
    megatron_path: str,
    output_path: str,
    num_layer: int = 32,
    hidden_size: int = 4096,
    num_query_heads: int = 32,
    num_kv_heads: int = 8,
    vocab_size: int = 73448,
    max_position_embeddings: int = 32768,
):
    """
    Convert MiniCPM Megatron checkpoint back to HuggingFace format
    """
    print(f"Converting Megatron to HF: {megatron_path} -> {output_path}")
    
    # Load Megatron checkpoint
    checkpoint_path = os.path.join(megatron_path, "mp_rank_00/model_optim_rng.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    megatron_state_dict = checkpoint["model"]
    
    print(f"Loaded Megatron checkpoint with {len(megatron_state_dict)} keys")
    
    # Create HF model config
    config = AutoConfig.from_pretrained("openbmb/MiniCPM4-8B", trust_remote_code=True)
    
    # Create HF model
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    
    # Convert state dict
    hf_state_dict = {}
    
    # Embedding
    if "embedding.word_embeddings.weight" in megatron_state_dict:
        hf_state_dict["model.embed_tokens.weight"] = megatron_state_dict["embedding.word_embeddings.weight"]
    
    # Final layer norm
    if "decoder.final_layernorm.weight" in megatron_state_dict:
        hf_state_dict["model.norm.weight"] = megatron_state_dict["decoder.final_layernorm.weight"]
    
    # Output layer
    if "output_layer.weight" in megatron_state_dict:
        hf_state_dict["lm_head.weight"] = megatron_state_dict["output_layer.weight"]
    
    # Process each layer
    for layer_idx in range(num_layer):
        print(f"Processing layer {layer_idx}")
        
        # Input layer norm
        hf_state_dict[f"model.layers.{layer_idx}.input_layernorm.weight"] = megatron_state_dict[
            f"decoder.layers.{layer_idx}.self_attention.linear_qkv.layer_norm_weight"
        ]
        
        # Post attention layer norm
        hf_state_dict[f"model.layers.{layer_idx}.post_attention_layernorm.weight"] = megatron_state_dict[
            f"decoder.layers.{layer_idx}.mlp.linear_fc1.layer_norm_weight"
        ]
        
        # Attention weights
        qkv_weight = megatron_state_dict[f"decoder.layers.{layer_idx}.self_attention.linear_qkv.weight"]
        o_weight = megatron_state_dict[f"decoder.layers.{layer_idx}.self_attention.linear_proj.weight"]
        
        # Split QKV weights back to separate Q, K, V
        head_dim = hidden_size // num_query_heads
        kv_head_dim = hidden_size // num_kv_heads
        num_query_heads_per_group = num_query_heads // num_kv_heads
        
        # Calculate splits
        q_splits = []
        k_splits = []
        v_splits = []
        
        current_pos = 0
        for kv_idx in range(num_kv_heads):
            # Add query heads for this KV head
            for q_idx in range(num_query_heads_per_group):
                q_splits.append(qkv_weight[current_pos:current_pos + head_dim])
                current_pos += head_dim
            # Add K and V heads
            k_splits.append(qkv_weight[current_pos:current_pos + kv_head_dim])
            current_pos += kv_head_dim
            v_splits.append(qkv_weight[current_pos:current_pos + kv_head_dim])
            current_pos += kv_head_dim
        
        # Concatenate back to original format
        hf_state_dict[f"model.layers.{layer_idx}.self_attn.q_proj.weight"] = torch.cat(q_splits, dim=0)
        hf_state_dict[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = torch.cat(k_splits, dim=0)
        hf_state_dict[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = torch.cat(v_splits, dim=0)
        hf_state_dict[f"model.layers.{layer_idx}.self_attn.o_proj.weight"] = o_weight
        
        # MLP weights
        fc1_weight = megatron_state_dict[f"decoder.layers.{layer_idx}.mlp.linear_fc1.weight"]
        fc2_weight = megatron_state_dict[f"decoder.layers.{layer_idx}.mlp.linear_fc2.weight"]
        
        # Split fc1 back to gate and up projections
        split_size = fc1_weight.shape[0] // 2
        hf_state_dict[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = fc1_weight[:split_size]
        hf_state_dict[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = fc1_weight[split_size:]
        hf_state_dict[f"model.layers.{layer_idx}.mlp.down_proj.weight"] = fc2_weight
    
    # Load state dict into model
    model.load_state_dict(hf_state_dict, strict=False)
    
    # Save the model
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    
    print(f"Conversion completed: {output_path}")
    return model

if __name__ == "__main__":
    reverse_convert_minicpm_megatron_to_hf(
        megatron_path="./test_output_minicpm4_8b_megatron",
        output_path="./test_output_minicpm4_8b_hf_reverse"
    ) 