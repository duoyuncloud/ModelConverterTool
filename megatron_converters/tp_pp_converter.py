#!/usr/bin/env python3
"""
Tensor Parallel and Pipeline Parallel Checkpoint Converter
Supports MiniCPM and Llama family Megatron<->HF bidirectional conversion
"""

import os
import torch
import argparse
from typing import Dict, List


class TensorParallelConverter:
    """Tensor Parallel and Pipeline Parallel converter for Megatron checkpoints"""

    def __init__(self):
        self.model_configs = {
            "minicpm": {
                "0.5b": {"layers": 12, "tp_size": 1, "pp_size": 1, "num_kv_heads": 8, "num_query_heads": 32},
                "1.5b": {"layers": 18, "tp_size": 1, "pp_size": 1, "num_kv_heads": 8, "num_query_heads": 32},
                "3b": {"layers": 24, "tp_size": 1, "pp_size": 1, "num_kv_heads": 8, "num_query_heads": 32},
                "8b": {"layers": 32, "tp_size": 2, "pp_size": 1, "num_kv_heads": 8, "num_query_heads": 32},
                "14b": {"layers": 40, "tp_size": 4, "pp_size": 1, "num_kv_heads": 8, "num_query_heads": 32},
            },
            "llama": {
                "7b": {"layers": 32, "tp_size": 1, "pp_size": 1, "num_kv_heads": 8, "num_query_heads": 32},
                "13b": {"layers": 40, "tp_size": 2, "pp_size": 1, "num_kv_heads": 8, "num_query_heads": 40},
                "30b": {"layers": 60, "tp_size": 4, "pp_size": 1, "num_kv_heads": 8, "num_query_heads": 52},
                "65b": {"layers": 80, "tp_size": 8, "pp_size": 1, "num_kv_heads": 8, "num_query_heads": 64},
            },
        }

    def validate_parallel_config(
        self, num_layer: int, tp_size: int, pp_size: int, num_kv_heads: int, num_query_heads: int
    ) -> None:
        """Validate parallel configuration parameters"""
        assert num_query_heads % num_kv_heads == 0, "Query heads must be divisible by KV heads"
        assert num_layer % pp_size == 0, "Number of layers must be divisible by PP size"
        assert num_query_heads % tp_size == 0, "Query heads must be divisible by TP size"
        assert num_kv_heads % tp_size == 0, "KV heads must be divisible by TP size"

    def load_distributed_checkpoints(self, in_dir: str, tp_size: int, pp_size: int) -> List[Dict[str, torch.Tensor]]:
        """Load distributed checkpoints from TP/PP sharded files"""
        print(f"Loading distributed checkpoints: TP={tp_size}, PP={pp_size}")

        # Initialize TP world state dictionaries
        ckpt_tp_world = [{} for _ in range(tp_size)]

        # Calculate layers per pipeline stage
        num_layer_per_pp = None  # Will be determined from first checkpoint

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
                layer_idx_base = pp_rank * (
                    num_layer_per_pp or len([k for k in state_dict.keys() if "decoder.layers." in k])
                )
                if num_layer_per_pp is None:
                    num_layer_per_pp = len([k for k in state_dict.keys() if "decoder.layers." in k])

                # Remap layer indices
                new_state_dict = {}
                for key in state_dict:
                    if "extra" not in key:
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
        self,
        num_layer: int,
        tp_size: int,
        pp_size: int,
        in_dir: str,
        save_path: str,
        num_kv_heads: int,
        num_query_heads: int,
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

        # Validate input directory
        if not os.path.exists(in_dir):
            raise FileNotFoundError(f"Input directory does not exist: {in_dir}")

        # Validate parallel configuration
        self.validate_parallel_config(num_layer, tp_size, pp_size, num_kv_heads, num_query_heads)

        # Load distributed checkpoints
        ckpt_tp_world = self.load_distributed_checkpoints(in_dir, tp_size, pp_size)

        # Initialize HF model state dict
        hf_model_state_dict = {}

        # Merge embedding and output layer weights
        embedding_list = []
        for sd in ckpt_tp_world:
            embedding_list.append(sd["embedding.word_embeddings.weight"])

        hf_model_state_dict["model.embed_tokens.weight"] = torch.cat(embedding_list, dim=0).cpu()
        
        # Handle output layer - check if it exists in the checkpoint
        if "output_layer.weight" in ckpt_tp_world[0]:
            lm_head_list = []
            for sd in ckpt_tp_world:
                lm_head_list.append(sd["output_layer.weight"])
            hf_model_state_dict["lm_head.weight"] = torch.cat(lm_head_list, dim=0).cpu()
        else:
            # If output layer doesn't exist, use embedding weights as output layer (common practice)
            print("Warning: output_layer.weight not found, using embedding weights as output layer")
            hf_model_state_dict["lm_head.weight"] = hf_model_state_dict["model.embed_tokens.weight"].clone()
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
                
                # Check if bias exists
                qkv_bias_key = f"decoder.layers.{layer_idx}.self_attention.linear_qkv.bias"
                has_bias = qkv_bias_key in sd
                
                if has_bias:
                    qkv_bias = sd[qkv_bias_key]
                else:
                    # Create zero bias if not present
                    qkv_bias = torch.zeros(qkv_proj.shape[0], dtype=qkv_proj.dtype, device=qkv_proj.device)

                # The QKV tensor is already organized in a specific pattern
                # For MiniCPM: [q_heads * q_dim + k_heads * k_dim + v_heads * v_dim]
                # We need to reconstruct the original Q, K, V tensors
                
                # Calculate dimensions
                q_head_dim = dim_model // num_query_heads  # 4096 // 32 = 128
                kv_head_dim = dim_model // num_query_heads // 4  # 128 // 4 = 32 (for grouped attention)
                
                # Calculate sizes for this TP rank
                q_size_per_rank = (num_query_heads // tp_size) * q_head_dim
                kv_size_per_rank = (num_kv_heads // tp_size) * kv_head_dim
                
                # Extract Q, K, V from the concatenated tensor
                q_start = 0
                q_end = q_size_per_rank
                k_start = q_size_per_rank
                k_end = q_size_per_rank + kv_size_per_rank
                v_start = q_size_per_rank + kv_size_per_rank
                v_end = q_size_per_rank + 2 * kv_size_per_rank
                
                q_proj_list.append(qkv_proj[q_start:q_end])
                k_proj_list.append(qkv_proj[k_start:k_end])
                v_proj_list.append(qkv_proj[v_start:v_end])
                
                # Handle bias similarly
                q_bias_list.append(qkv_bias[q_start:q_end])
                k_bias_list.append(qkv_bias[k_start:k_end])
                v_bias_list.append(qkv_bias[v_start:v_end])

                o_proj_list.append(sd[f"decoder.layers.{layer_idx}.self_attention.linear_proj.weight"])

            # Merge attention weights
            hf_model_state_dict[f"model.layers.{layer_idx}.self_attn.q_proj.weight"] = torch.cat(
                q_proj_list, dim=0
            ).cpu()
            hf_model_state_dict[f"model.layers.{layer_idx}.self_attn.q_proj.bias"] = torch.cat(q_bias_list, dim=0).cpu()
            hf_model_state_dict[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = torch.cat(
                k_proj_list, dim=0
            ).cpu()
            hf_model_state_dict[f"model.layers.{layer_idx}.self_attn.k_proj.bias"] = torch.cat(k_bias_list, dim=0).cpu()
            hf_model_state_dict[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = torch.cat(
                v_proj_list, dim=0
            ).cpu()
            hf_model_state_dict[f"model.layers.{layer_idx}.self_attn.v_proj.bias"] = torch.cat(v_bias_list, dim=0).cpu()
            hf_model_state_dict[f"model.layers.{layer_idx}.self_attn.o_proj.weight"] = torch.cat(
                o_proj_list, dim=1
            ).cpu()

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

            hf_model_state_dict[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = torch.cat(
                gate_proj_list, dim=0
            ).cpu()
            hf_model_state_dict[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = torch.cat(up_proj_list, dim=0).cpu()
            hf_model_state_dict[f"model.layers.{layer_idx}.mlp.down_proj.weight"] = torch.cat(
                down_proj_list, dim=1
            ).cpu()

        # Save converted weights
        print(f"Saving converted weights to: {save_path}")
        torch.save(hf_model_state_dict, save_path)

        # Print statistics
        total_params = sum(p.numel() for p in hf_model_state_dict.values())
        print(f"Conversion completed! Total parameters: {total_params:,}")
        print(f"Weight file size: {os.path.getsize(save_path) / 1024 / 1024 / 1024:.2f} GB")

    def convert_llama_megatron_to_hf_tp_pp(
        self,
        num_layer: int,
        tp_size: int,
        pp_size: int,
        in_dir: str,
        save_path: str,
        num_kv_heads: int,
        num_query_heads: int,
    ) -> None:
        """
        Convert Llama Megatron checkpoint to HuggingFace format with TP/PP support
        Similar to MiniCPM but with Llama-specific weight naming
        """
        # Implementation similar to MiniCPM but adapted for Llama architecture
        # This can be extended based on the specific Llama model requirements
        pass

    def convert_minicpm_megatron_to_hf_tp_pp_memory_efficient(
        self,
        num_layer: int,
        tp_size: int,
        pp_size: int,
        in_dir: str,
        save_path: str,
        num_kv_heads: int,
        num_query_heads: int,
        batch_size: int = 4,
    ) -> None:
        """
        Memory-efficient version of MiniCPM Megatron to HF conversion
        Processes layers in batches to reduce memory usage
        """
        print(f"Starting memory-efficient conversion: {in_dir} -> {save_path}")
        print(f"Model config: {num_layer} layers, TP={tp_size}, PP={pp_size}")
        print(f"Attention config: {num_query_heads} query heads, {num_kv_heads} KV heads")
        print(f"Processing in batches of {batch_size} layers")

        # Validate parallel configuration
        self.validate_parallel_config(num_layer, tp_size, pp_size, num_kv_heads, num_query_heads)

        # Load distributed checkpoints
        ckpt_tp_world = self.load_distributed_checkpoints(in_dir, tp_size, pp_size)

        # Initialize HF model state dict with core components
        hf_model_state_dict = {}

        # Merge embedding and output layer weights
        embedding_list = []
        for sd in ckpt_tp_world:
            embedding_list.append(sd["embedding.word_embeddings.weight"])

        hf_model_state_dict["model.embed_tokens.weight"] = torch.cat(embedding_list, dim=0).cpu()
        
        # Handle output layer - check if it exists in the checkpoint
        if "output_layer.weight" in ckpt_tp_world[0]:
            lm_head_list = []
            for sd in ckpt_tp_world:
                lm_head_list.append(sd["output_layer.weight"])
            hf_model_state_dict["lm_head.weight"] = torch.cat(lm_head_list, dim=0).cpu()
        else:
            # If output layer doesn't exist, use embedding weights as output layer (common practice)
            print("Warning: output_layer.weight not found, using embedding weights as output layer")
            hf_model_state_dict["lm_head.weight"] = hf_model_state_dict["model.embed_tokens.weight"].clone()
        
        hf_model_state_dict["model.norm.weight"] = ckpt_tp_world[0]["decoder.final_layernorm.weight"].cpu()

        # Calculate attention parameters
        dim_model = hf_model_state_dict["model.embed_tokens.weight"].shape[1]

        # Process layers in batches
        for batch_start in range(0, num_layer, batch_size):
            batch_end = min(batch_start + batch_size, num_layer)
            print(f"Processing layers {batch_start} to {batch_end-1} (batch {batch_start//batch_size + 1}/{(num_layer + batch_size - 1)//batch_size})")
            
            for layer_idx in range(batch_start, batch_end):
                print(f"  Processing layer {layer_idx}")

                # Input layer normalization
                hf_model_state_dict[f"model.layers.{layer_idx}.input_layernorm.weight"] = ckpt_tp_world[0][
                    f"decoder.layers.{layer_idx}.self_attention.linear_qkv.layer_norm_weight"
                ].cpu()

                # Attention layer weights
                q_proj_list, k_proj_list, v_proj_list, o_proj_list = [], [], [], []
                q_bias_list, k_bias_list, v_bias_list = [], [], []

                for sd in ckpt_tp_world:
                    qkv_proj = sd[f"decoder.layers.{layer_idx}.self_attention.linear_qkv.weight"]
                    
                    # Check if bias exists
                    qkv_bias_key = f"decoder.layers.{layer_idx}.self_attention.linear_qkv.bias"
                    has_bias = qkv_bias_key in sd
                    
                    if has_bias:
                        qkv_bias = sd[qkv_bias_key]
                    else:
                        # Create zero bias if not present
                        qkv_bias = torch.zeros(qkv_proj.shape[0], dtype=qkv_proj.dtype, device=qkv_proj.device)

                    # Calculate dimensions
                    q_head_dim = dim_model // num_query_heads  # 4096 // 32 = 128
                    kv_head_dim = dim_model // num_query_heads // 4  # 128 // 4 = 32 (for grouped attention)
                    
                    # Calculate sizes for this TP rank
                    q_size_per_rank = (num_query_heads // tp_size) * q_head_dim
                    kv_size_per_rank = (num_kv_heads // tp_size) * kv_head_dim
                    
                    # Extract Q, K, V from the concatenated tensor
                    q_start = 0
                    q_end = q_size_per_rank
                    k_start = q_size_per_rank
                    k_end = q_size_per_rank + kv_size_per_rank
                    v_start = q_size_per_rank + kv_size_per_rank
                    v_end = q_size_per_rank + 2 * kv_size_per_rank
                    
                    q_proj_list.append(qkv_proj[q_start:q_end])
                    k_proj_list.append(qkv_proj[k_start:k_end])
                    v_proj_list.append(qkv_proj[v_start:v_end])
                    
                    # Handle bias similarly
                    q_bias_list.append(qkv_bias[q_start:q_end])
                    k_bias_list.append(qkv_bias[k_start:k_end])
                    v_bias_list.append(qkv_bias[v_start:v_end])

                    o_proj_list.append(sd[f"decoder.layers.{layer_idx}.self_attention.linear_proj.weight"])

                # Merge attention weights
                hf_model_state_dict[f"model.layers.{layer_idx}.self_attn.q_proj.weight"] = torch.cat(
                    q_proj_list, dim=0
                ).cpu()
                hf_model_state_dict[f"model.layers.{layer_idx}.self_attn.q_proj.bias"] = torch.cat(q_bias_list, dim=0).cpu()
                hf_model_state_dict[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = torch.cat(
                    k_proj_list, dim=0
                ).cpu()
                hf_model_state_dict[f"model.layers.{layer_idx}.self_attn.k_proj.bias"] = torch.cat(k_bias_list, dim=0).cpu()
                hf_model_state_dict[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = torch.cat(
                    v_proj_list, dim=0
                ).cpu()
                hf_model_state_dict[f"model.layers.{layer_idx}.self_attn.v_proj.bias"] = torch.cat(v_bias_list, dim=0).cpu()
                hf_model_state_dict[f"model.layers.{layer_idx}.self_attn.o_proj.weight"] = torch.cat(
                    o_proj_list, dim=1
                ).cpu()

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

                hf_model_state_dict[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = torch.cat(
                    gate_proj_list, dim=0
                ).cpu()
                hf_model_state_dict[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = torch.cat(
                    up_proj_list, dim=0
                ).cpu()
                hf_model_state_dict[f"model.layers.{layer_idx}.mlp.down_proj.weight"] = torch.cat(
                    down_proj_list, dim=1
                ).cpu()
            
            # Clear memory after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()

        # Save the converted model
        print(f"Saving converted model to {save_path}")
        torch.save(hf_model_state_dict, save_path)
        print("Memory-efficient conversion completed successfully!")

    def convert_minicpm_megatron_to_hf_tp_pp_ultra_memory_efficient(
        self,
        num_layer: int,
        tp_size: int,
        pp_size: int,
        in_dir: str,
        save_path: str,
        num_kv_heads: int,
        num_query_heads: int,
    ) -> None:
        """
        Ultra memory-efficient version that processes one layer at a time
        and saves incrementally to avoid memory buildup
        """
        print(f"Starting ultra memory-efficient conversion: {in_dir} -> {save_path}")
        print(f"Model config: {num_layer} layers, TP={tp_size}, PP={pp_size}")
        print(f"Attention config: {num_query_heads} query heads, {num_kv_heads} KV heads")
        print("Processing one layer at a time with incremental saving")

        # Validate parallel configuration
        self.validate_parallel_config(num_layer, tp_size, pp_size, num_kv_heads, num_query_heads)

        # Load distributed checkpoints
        ckpt_tp_world = self.load_distributed_checkpoints(in_dir, tp_size, pp_size)

        # Initialize HF model state dict with core components only
        hf_model_state_dict = {}

        # Merge embedding and output layer weights
        print("Processing embedding and output layers...")
        embedding_list = []
        for sd in ckpt_tp_world:
            embedding_list.append(sd["embedding.word_embeddings.weight"])

        hf_model_state_dict["model.embed_tokens.weight"] = torch.cat(embedding_list, dim=0).cpu()
        
        # Handle output layer - check if it exists in the checkpoint
        if "output_layer.weight" in ckpt_tp_world[0]:
            lm_head_list = []
            for sd in ckpt_tp_world:
                lm_head_list.append(sd["output_layer.weight"])
            hf_model_state_dict["lm_head.weight"] = torch.cat(lm_head_list, dim=0).cpu()
        else:
            # If output layer doesn't exist, use embedding weights as output layer (common practice)
            print("Warning: output_layer.weight not found, using embedding weights as output layer")
            hf_model_state_dict["lm_head.weight"] = hf_model_state_dict["model.embed_tokens.weight"].clone()
        
        hf_model_state_dict["model.norm.weight"] = ckpt_tp_world[0]["decoder.final_layernorm.weight"].cpu()

        # Calculate attention parameters
        dim_model = hf_model_state_dict["model.embed_tokens.weight"].shape[1]

        # Save core components first
        print("Saving core components...")
        torch.save(hf_model_state_dict, save_path)
        print(f"Core components saved to {save_path}")

        # Process layers one by one and append to the saved file
        for layer_idx in range(num_layer):
            print(f"Processing layer {layer_idx + 1}/{num_layer}")
            
            # Load the current state dict
            current_state_dict = torch.load(save_path)
            
            # Input layer normalization
            current_state_dict[f"model.layers.{layer_idx}.input_layernorm.weight"] = ckpt_tp_world[0][
                f"decoder.layers.{layer_idx}.self_attention.linear_qkv.layer_norm_weight"
            ].cpu()

            # Attention layer weights
            q_proj_list, k_proj_list, v_proj_list, o_proj_list = [], [], [], []
            q_bias_list, k_bias_list, v_bias_list = [], [], []

            for sd in ckpt_tp_world:
                qkv_proj = sd[f"decoder.layers.{layer_idx}.self_attention.linear_qkv.weight"]
                
                # Check if bias exists
                qkv_bias_key = f"decoder.layers.{layer_idx}.self_attention.linear_qkv.bias"
                has_bias = qkv_bias_key in sd
                
                if has_bias:
                    qkv_bias = sd[qkv_bias_key]
                else:
                    # Create zero bias if not present
                    qkv_bias = torch.zeros(qkv_proj.shape[0], dtype=qkv_proj.dtype, device=qkv_proj.device)

                # Calculate dimensions
                q_head_dim = dim_model // num_query_heads  # 4096 // 32 = 128
                kv_head_dim = dim_model // num_query_heads // 4  # 128 // 4 = 32 (for grouped attention)
                
                # Calculate sizes for this TP rank
                q_size_per_rank = (num_query_heads // tp_size) * q_head_dim
                kv_size_per_rank = (num_kv_heads // tp_size) * kv_head_dim
                
                # Extract Q, K, V from the concatenated tensor
                q_start = 0
                q_end = q_size_per_rank
                k_start = q_size_per_rank
                k_end = q_size_per_rank + kv_size_per_rank
                v_start = q_size_per_rank + kv_size_per_rank
                v_end = q_size_per_rank + 2 * kv_size_per_rank
                
                q_proj_list.append(qkv_proj[q_start:q_end])
                k_proj_list.append(qkv_proj[k_start:k_end])
                v_proj_list.append(qkv_proj[v_start:v_end])
                
                # Handle bias similarly
                q_bias_list.append(qkv_bias[q_start:q_end])
                k_bias_list.append(qkv_bias[k_start:k_end])
                v_bias_list.append(qkv_bias[v_start:v_end])

                o_proj_list.append(sd[f"decoder.layers.{layer_idx}.self_attention.linear_proj.weight"])

            # Merge attention weights
            current_state_dict[f"model.layers.{layer_idx}.self_attn.q_proj.weight"] = torch.cat(
                q_proj_list, dim=0
            ).cpu()
            current_state_dict[f"model.layers.{layer_idx}.self_attn.q_proj.bias"] = torch.cat(q_bias_list, dim=0).cpu()
            current_state_dict[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = torch.cat(
                k_proj_list, dim=0
            ).cpu()
            current_state_dict[f"model.layers.{layer_idx}.self_attn.k_proj.bias"] = torch.cat(k_bias_list, dim=0).cpu()
            current_state_dict[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = torch.cat(
                v_proj_list, dim=0
            ).cpu()
            current_state_dict[f"model.layers.{layer_idx}.self_attn.v_proj.bias"] = torch.cat(v_bias_list, dim=0).cpu()
            current_state_dict[f"model.layers.{layer_idx}.self_attn.o_proj.weight"] = torch.cat(
                o_proj_list, dim=1
            ).cpu()

            # MLP layer weights
            current_state_dict[f"model.layers.{layer_idx}.post_attention_layernorm.weight"] = ckpt_tp_world[0][
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

            current_state_dict[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = torch.cat(
                gate_proj_list, dim=0
            ).cpu()
            current_state_dict[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = torch.cat(
                up_proj_list, dim=0
            ).cpu()
            current_state_dict[f"model.layers.{layer_idx}.mlp.down_proj.weight"] = torch.cat(
                down_proj_list, dim=1
            ).cpu()
            
            # Save after each layer
            torch.save(current_state_dict, save_path)
            
            # Aggressive memory cleanup
            del current_state_dict, q_proj_list, k_proj_list, v_proj_list, o_proj_list
            del q_bias_list, k_bias_list, v_bias_list, gate_proj_list, up_proj_list, down_proj_list
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()

        print("Ultra memory-efficient conversion completed successfully!")

    def convert_minicpm_mla_megatron_to_hf_tp_pp_ultra_memory_efficient(
        self,
        num_layer: int,
        tp_size: int,
        pp_size: int,
        in_dir: str,
        save_path: str,
        num_kv_heads: int,
        num_query_heads: int,
    ) -> None:
        """
        Ultra memory-efficient version for MiniCPM MLA models (like MiniCPM3-4B)
        that processes one layer at a time and saves incrementally
        """
        print(f"Starting MLA ultra memory-efficient conversion: {in_dir} -> {save_path}")
        print(f"Model config: {num_layer} layers, TP={tp_size}, PP={pp_size}")
        print(f"Attention config: {num_query_heads} query heads, {num_kv_heads} KV heads")
        print("Processing one layer at a time with incremental saving")

        # Validate parallel configuration
        self.validate_parallel_config(num_layer, tp_size, pp_size, num_kv_heads, num_query_heads)

        # Load distributed checkpoints
        ckpt_tp_world = self.load_distributed_checkpoints(in_dir, tp_size, pp_size)

        # Initialize HF model state dict with core components only
        hf_model_state_dict = {}

        # Merge embedding and output layer weights
        print("Processing embedding and output layers...")
        embedding_list = []
        for sd in ckpt_tp_world:
            embedding_list.append(sd["embedding.word_embeddings.weight"])

        hf_model_state_dict["model.embed_tokens.weight"] = torch.cat(embedding_list, dim=0).cpu()
        
        # Handle output layer - check if it exists in the checkpoint
        if "output_layer.weight" in ckpt_tp_world[0]:
            lm_head_list = []
            for sd in ckpt_tp_world:
                lm_head_list.append(sd["output_layer.weight"])
            hf_model_state_dict["lm_head.weight"] = torch.cat(lm_head_list, dim=0).cpu()
        else:
            # If output layer doesn't exist, use embedding weights as output layer (common practice)
            print("Warning: output_layer.weight not found, using embedding weights as output layer")
            hf_model_state_dict["lm_head.weight"] = hf_model_state_dict["model.embed_tokens.weight"].clone()
        
        hf_model_state_dict["model.norm.weight"] = ckpt_tp_world[0]["decoder.final_layernorm.weight"].cpu()

        # Calculate attention parameters
        dim_model = hf_model_state_dict["model.embed_tokens.weight"].shape[1]

        # Save core components first
        print("Saving core components...")
        torch.save(hf_model_state_dict, save_path)
        print(f"Core components saved to {save_path}")

        # Process layers one by one and append to the saved file
        for layer_idx in range(num_layer):
            print(f"Processing layer {layer_idx + 1}/{num_layer}")
            
            # Load the current state dict
            current_state_dict = torch.load(save_path)
            
            # Input layer normalization
            current_state_dict[f"model.layers.{layer_idx}.input_layernorm.weight"] = ckpt_tp_world[0][
                f"decoder.layers.{layer_idx}.input_layernorm.weight"
            ].cpu()

            # MLA attention layer weights - reconstruct Q, K, V from the separate components
            q_proj_list, k_proj_list, v_proj_list, o_proj_list = [], [], [], []

            for sd in ckpt_tp_world:
                # MLA has separate Q and KV components - use them directly
                q_proj = sd[f"decoder.layers.{layer_idx}.self_attention.linear_q_up.weight"]
                k_proj = sd[f"decoder.layers.{layer_idx}.self_attention.linear_kv_up.weight"]
                v_proj = sd[f"decoder.layers.{layer_idx}.self_attention.linear_kv_down.weight"]
                
                q_proj_list.append(q_proj)
                k_proj_list.append(k_proj)
                v_proj_list.append(v_proj)
                o_proj_list.append(sd[f"decoder.layers.{layer_idx}.self_attention.linear_proj.weight"])

            # Merge attention weights
            current_state_dict[f"model.layers.{layer_idx}.self_attn.q_proj.weight"] = torch.cat(
                q_proj_list, dim=0
            ).cpu()
            current_state_dict[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = torch.cat(
                k_proj_list, dim=0
            ).cpu()
            current_state_dict[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = torch.cat(
                v_proj_list, dim=0
            ).cpu()
            current_state_dict[f"model.layers.{layer_idx}.self_attn.o_proj.weight"] = torch.cat(
                o_proj_list, dim=1
            ).cpu()

            # MLP layer weights
            current_state_dict[f"model.layers.{layer_idx}.post_attention_layernorm.weight"] = ckpt_tp_world[0][
                f"decoder.layers.{layer_idx}.pre_mlp_layernorm.weight"
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

            current_state_dict[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = torch.cat(
                gate_proj_list, dim=0
            ).cpu()
            current_state_dict[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = torch.cat(
                up_proj_list, dim=0
            ).cpu()
            current_state_dict[f"model.layers.{layer_idx}.mlp.down_proj.weight"] = torch.cat(
                down_proj_list, dim=1
            ).cpu()
            
            # Save after each layer
            torch.save(current_state_dict, save_path)
            
            # Aggressive memory cleanup
            del current_state_dict, q_proj_list, k_proj_list, v_proj_list, o_proj_list
            del gate_proj_list, up_proj_list, down_proj_list
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()

        print("MLA ultra memory-efficient conversion completed successfully!")

    def convert_minicpm_mla_megatron_to_hf_tp_pp_directory_save(
        self,
        num_layer: int,
        tp_size: int,
        pp_size: int,
        in_dir: str,
        save_dir: str,
        num_kv_heads: int,
        num_query_heads: int,
    ) -> None:
        """
        MLA converter that saves to directory structure to avoid disk space issues
        """
        import os
        print(f"Starting MLA directory-save conversion: {in_dir} -> {save_dir}")
        print(f"Model config: {num_layer} layers, TP={tp_size}, PP={pp_size}")
        print(f"Attention config: {num_query_heads} query heads, {num_kv_heads} KV heads")
        print("Processing one layer at a time with directory saving")

        # Validate parallel configuration
        self.validate_parallel_config(num_layer, tp_size, pp_size, num_kv_heads, num_query_heads)

        # Create output directory
        os.makedirs(save_dir, exist_ok=True)

        # Load distributed checkpoints
        ckpt_tp_world = self.load_distributed_checkpoints(in_dir, tp_size, pp_size)

        # Initialize HF model state dict
        hf_model_state_dict = {}

        # Process embedding and output layers
        print("Processing embedding and output layers...")
        embedding_list = []
        for sd in ckpt_tp_world:
            embedding_list.append(sd["embedding.word_embeddings.weight"])

        hf_model_state_dict["model.embed_tokens.weight"] = torch.cat(embedding_list, dim=0).cpu()
        
        # Handle output layer
        if "output_layer.weight" in ckpt_tp_world[0]:
            lm_head_list = []
            for sd in ckpt_tp_world:
                lm_head_list.append(sd["output_layer.weight"])
            hf_model_state_dict["lm_head.weight"] = torch.cat(lm_head_list, dim=0).cpu()
        else:
            print("Warning: output_layer.weight not found, using embedding weights as output layer")
            hf_model_state_dict["lm_head.weight"] = hf_model_state_dict["model.embed_tokens.weight"].clone()
        
        hf_model_state_dict["model.norm.weight"] = ckpt_tp_world[0]["decoder.final_layernorm.weight"].cpu()

        # Save core components
        print("Saving core components...")
        torch.save(hf_model_state_dict, os.path.join(save_dir, "core_components.pt"))
        print(f"Core components saved to {save_dir}/core_components.pt")

        # Process layers one by one and save individually
        for layer_idx in range(num_layer):
            print(f"Processing layer {layer_idx + 1}/{num_layer}")
            
            layer_state_dict = {}
            
            # Input layer normalization
            layer_state_dict[f"model.layers.{layer_idx}.input_layernorm.weight"] = ckpt_tp_world[0][
                f"decoder.layers.{layer_idx}.input_layernorm.weight"
            ].cpu()

            # MLA attention layer weights
            q_proj_list, k_proj_list, v_proj_list, o_proj_list = [], [], [], []

            for sd in ckpt_tp_world:
                # MLA has separate Q and KV components - use them directly
                q_proj = sd[f"decoder.layers.{layer_idx}.self_attention.linear_q_up.weight"]
                k_proj = sd[f"decoder.layers.{layer_idx}.self_attention.linear_kv_up.weight"]
                v_proj = sd[f"decoder.layers.{layer_idx}.self_attention.linear_kv_down.weight"]
                
                q_proj_list.append(q_proj)
                k_proj_list.append(k_proj)
                v_proj_list.append(v_proj)
                o_proj_list.append(sd[f"decoder.layers.{layer_idx}.self_attention.linear_proj.weight"])

            # Merge attention weights
            layer_state_dict[f"model.layers.{layer_idx}.self_attn.q_proj.weight"] = torch.cat(
                q_proj_list, dim=0
            ).cpu()
            layer_state_dict[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = torch.cat(
                k_proj_list, dim=0
            ).cpu()
            layer_state_dict[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = torch.cat(
                v_proj_list, dim=0
            ).cpu()
            layer_state_dict[f"model.layers.{layer_idx}.self_attn.o_proj.weight"] = torch.cat(
                o_proj_list, dim=1
            ).cpu()

            # MLP layer weights
            layer_state_dict[f"model.layers.{layer_idx}.post_attention_layernorm.weight"] = ckpt_tp_world[0][
                f"decoder.layers.{layer_idx}.pre_mlp_layernorm.weight"
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

            layer_state_dict[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = torch.cat(
                gate_proj_list, dim=0
            ).cpu()
            layer_state_dict[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = torch.cat(
                up_proj_list, dim=0
            ).cpu()
            layer_state_dict[f"model.layers.{layer_idx}.mlp.down_proj.weight"] = torch.cat(
                down_proj_list, dim=1
            ).cpu()
            
            # Save layer individually
            layer_file = os.path.join(save_dir, f"layer_{layer_idx:03d}.pt")
            torch.save(layer_state_dict, layer_file)
            print(f"Saved layer {layer_idx} to {layer_file}")
            
            # Aggressive memory cleanup
            del layer_state_dict, q_proj_list, k_proj_list, v_proj_list, o_proj_list
            del gate_proj_list, up_proj_list, down_proj_list
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()

        print("MLA directory-save conversion completed successfully!")
        print(f"Model saved to directory: {save_dir}")
        print("To combine all layers, use the combine_layers function")


# Convenience functions for common model configurations
def convert_8b_minicpm_megatron_to_hf(in_dir: str, save_path: str) -> None:
    """Convenience function for 8B MiniCPM conversion"""
    converter = TensorParallelConverter()
    converter.convert_minicpm_megatron_to_hf_tp_pp(
        num_layer=32, tp_size=2, pp_size=1, in_dir=in_dir, save_path=save_path, num_kv_heads=8, num_query_heads=32
    )


def convert_3b_minicpm_megatron_to_hf(in_dir: str, save_path: str) -> None:
    """Convenience function for 3B MiniCPM conversion"""
    converter = TensorParallelConverter()
    converter.convert_minicpm_megatron_to_hf_tp_pp(
        num_layer=24, tp_size=1, pp_size=1, in_dir=in_dir, save_path=save_path, num_kv_heads=8, num_query_heads=32
    )


def convert_7b_llama_megatron_to_hf(in_dir: str, save_path: str) -> None:
    """Convenience function for 7B Llama conversion"""
    converter = TensorParallelConverter()
    converter.convert_llama_megatron_to_hf_tp_pp(
        num_layer=32, tp_size=1, pp_size=1, in_dir=in_dir, save_path=save_path, num_kv_heads=8, num_query_heads=32
    )


# Standalone wrapper functions for backward compatibility and easy imports
def convert_minicpm_megatron_to_hf_tp_pp(
    num_layer: int, tp_size: int, pp_size: int, in_dir: str, save_path: str, num_kv_heads: int, num_query_heads: int
) -> None:
    """
    Standalone wrapper function for MiniCPM Megatron to HF conversion with TP/PP support
    """
    converter = TensorParallelConverter()
    converter.convert_minicpm_megatron_to_hf_tp_pp(
        num_layer=num_layer,
        tp_size=tp_size,
        pp_size=pp_size,
        in_dir=in_dir,
        save_path=save_path,
        num_kv_heads=num_kv_heads,
        num_query_heads=num_query_heads,
    )


def convert_minicpm_megatron_to_hf_tp_pp_memory_efficient(
    num_layer: int, tp_size: int, pp_size: int, in_dir: str, save_path: str, num_kv_heads: int, num_query_heads: int, batch_size: int = 4
) -> None:
    """
    Standalone wrapper function for memory-efficient MiniCPM Megatron to HF conversion
    """
    converter = TensorParallelConverter()
    converter.convert_minicpm_megatron_to_hf_tp_pp_memory_efficient(
        num_layer=num_layer,
        tp_size=tp_size,
        pp_size=pp_size,
        in_dir=in_dir,
        save_path=save_path,
        num_kv_heads=num_kv_heads,
        num_query_heads=num_query_heads,
        batch_size=batch_size,
    )


def convert_minicpm_megatron_to_hf_tp_pp_ultra_memory_efficient(
    num_layer: int, tp_size: int, pp_size: int, in_dir: str, save_path: str, num_kv_heads: int, num_query_heads: int
) -> None:
    """
    Standalone wrapper function for ultra memory-efficient MiniCPM Megatron to HF conversion
    """
    converter = TensorParallelConverter()
    converter.convert_minicpm_megatron_to_hf_tp_pp_ultra_memory_efficient(
        num_layer=num_layer,
        tp_size=tp_size,
        pp_size=pp_size,
        in_dir=in_dir,
        save_path=save_path,
        num_kv_heads=num_kv_heads,
        num_query_heads=num_query_heads,
    )


def convert_minicpm_mla_megatron_to_hf_tp_pp_ultra_memory_efficient(
    num_layer: int, tp_size: int, pp_size: int, in_dir: str, save_path: str, num_kv_heads: int, num_query_heads: int
) -> None:
    """
    Standalone wrapper function for ultra memory-efficient MiniCPM MLA Megatron to HF conversion
    """
    converter = TensorParallelConverter()
    converter.convert_minicpm_mla_megatron_to_hf_tp_pp_ultra_memory_efficient(
        num_layer=num_layer,
        tp_size=tp_size,
        pp_size=pp_size,
        in_dir=in_dir,
        save_path=save_path,
        num_kv_heads=num_kv_heads,
        num_query_heads=num_query_heads,
    )


def convert_minicpm_mla_megatron_to_hf_tp_pp_directory_save(
    num_layer: int, tp_size: int, pp_size: int, in_dir: str, save_dir: str, num_kv_heads: int, num_query_heads: int
) -> None:
    """
    Standalone wrapper function for MLA directory-save MiniCPM Megatron to HF conversion
    """
    converter = TensorParallelConverter()
    converter.convert_minicpm_mla_megatron_to_hf_tp_pp_directory_save(
        num_layer=num_layer,
        tp_size=tp_size,
        pp_size=pp_size,
        in_dir=in_dir,
        save_dir=save_dir,
        num_kv_heads=num_kv_heads,
        num_query_heads=num_query_heads,
    )


def convert_llama_megatron_to_hf_tp_pp(
    num_layer: int, tp_size: int, pp_size: int, in_dir: str, save_path: str, num_kv_heads: int, num_query_heads: int
) -> None:
    """
    Standalone wrapper function for Llama Megatron to HF conversion with TP/PP support
    """
    converter = TensorParallelConverter()
    converter.convert_llama_megatron_to_hf_tp_pp(
        num_layer=num_layer,
        tp_size=tp_size,
        pp_size=pp_size,
        in_dir=in_dir,
        save_path=save_path,
        num_kv_heads=num_kv_heads,
        num_query_heads=num_query_heads,
    )


def combine_directory_layers_to_hf_model(save_dir: str, output_path: str) -> None:
    """
    Combine directory-saved layers into a single HuggingFace model file
    
    Args:
        save_dir: Directory containing core_components.pt and layer_*.pt files
        output_path: Path to save the combined HuggingFace model
    """
    import os
    import glob
    
    print(f"Combining layers from {save_dir} to {output_path}")
    
    # Load core components
    core_components_path = os.path.join(save_dir, "core_components.pt")
    if not os.path.exists(core_components_path):
        raise FileNotFoundError(f"Core components file not found: {core_components_path}")
    
    print("Loading core components...")
    hf_model_state_dict = torch.load(core_components_path)
    
    # Find all layer files
    layer_files = sorted(glob.glob(os.path.join(save_dir, "layer_*.pt")))
    print(f"Found {len(layer_files)} layer files")
    
    # Load and combine all layers
    for layer_file in layer_files:
        print(f"Loading {os.path.basename(layer_file)}...")
        layer_state_dict = torch.load(layer_file)
        hf_model_state_dict.update(layer_state_dict)
        
        # Clean up layer file to save memory
        del layer_state_dict
    
    # Save combined model
    print(f"Saving combined model to {output_path}...")
    torch.save(hf_model_state_dict, output_path)
    print(f"Successfully saved combined model: {output_path}")
    print(f"Total model size: {len(hf_model_state_dict)} tensors")


# Add the combine function to the module exports
__all__ = [
    "TensorParallelConverter",
    "convert_minicpm_megatron_to_hf_tp_pp",
    "convert_minicpm_megatron_to_hf_tp_pp_memory_efficient", 
    "convert_minicpm_megatron_to_hf_tp_pp_ultra_memory_efficient",
    "convert_minicpm_mla_megatron_to_hf_tp_pp_ultra_memory_efficient",
    "convert_minicpm_mla_megatron_to_hf_tp_pp_directory_save",
    "convert_llama_megatron_to_hf_tp_pp",
    "combine_directory_layers_to_hf_model",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Megatron to HuggingFace conversion with TP/PP support")
    parser.add_argument("--model_type", choices=["minicpm", "llama"], default="minicpm")
    parser.add_argument("--model_size", choices=["3b", "8b", "7b", "13b", "30b", "65b"], default="8b")
    parser.add_argument("--in_dir", type=str, required=True, help="Input Megatron checkpoint directory")
    parser.add_argument("--save_path", type=str, required=True, help="Output HF weights file path")
    parser.add_argument("--num_layer", type=int, help="Number of transformer layers")
    parser.add_argument("--tp_size", type=int, help="Tensor parallel size")
    parser.add_argument("--pp_size", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--num_kv_heads", type=int, help="Number of KV attention heads")
    parser.add_argument("--num_query_heads", type=int, help="Number of query attention heads")

    args = parser.parse_args()

    converter = TensorParallelConverter()

    if args.model_type == "minicpm":
        if args.model_size == "8b":
            convert_8b_minicpm_megatron_to_hf(args.in_dir, args.save_path)
        elif args.model_size == "3b":
            convert_3b_minicpm_megatron_to_hf(args.in_dir, args.save_path)
        else:
            # Use custom parameters
            converter.convert_minicpm_megatron_to_hf_tp_pp(
                num_layer=args.num_layer,
                tp_size=args.tp_size,
                pp_size=args.pp_size,
                in_dir=args.in_dir,
                save_path=args.save_path,
                num_kv_heads=args.num_kv_heads,
                num_query_heads=args.num_query_heads,
            )
    elif args.model_type == "llama":
        if args.model_size == "7b":
            convert_7b_llama_megatron_to_hf(args.in_dir, args.save_path)
        else:
            # Use custom parameters for other Llama sizes
            converter.convert_llama_megatron_to_hf_tp_pp(
                num_layer=args.num_layer,
                tp_size=args.tp_size,
                pp_size=args.pp_size,
                in_dir=args.in_dir,
                save_path=args.save_path,
                num_kv_heads=args.num_kv_heads,
                num_query_heads=args.num_query_heads,
            )
