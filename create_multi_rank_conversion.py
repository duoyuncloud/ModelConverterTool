#!/usr/bin/env python3
"""
Create multi-rank tensor parallel conversion for proper reverse conversion
"""

import torch
import os
from megatron_converters.hf_to_megatron_minicpm import convert_hf_to_megatron_minicpm_main

def create_multi_rank_megatron_checkpoint(
    hf_model_path: str,
    output_dir: str,
    num_layer: int = 32,
    tp_size: int = 2,
    pp_size: int = 1,
    num_kv_heads: int = 8,
    num_query_heads: int = 32,
    use_mla: bool = False,
):
    """
    Create a proper multi-rank tensor parallel Megatron checkpoint
    """
    print(f"Creating multi-rank Megatron checkpoint: {hf_model_path} -> {output_dir}")
    print(f"Configuration: {num_layer} layers, TP={tp_size}, PP={pp_size}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate each tensor parallel rank
    for tp_rank in range(tp_size):
        print(f"\nGenerating rank {tp_rank}...")
        
        # Create rank-specific output directory
        if pp_size == 1:
            rank_output_dir = os.path.join(output_dir, f"mp_rank_{tp_rank:02d}")
        else:
            rank_output_dir = os.path.join(output_dir, f"mp_rank_{tp_rank:02d}_000")
        
        os.makedirs(rank_output_dir, exist_ok=True)
        
        # Convert for this specific rank
        convert_hf_to_megatron_minicpm_main(
            load_path=hf_model_path,
            num_layer=num_layer,
            tp_size=tp_size,
            tp_rank=tp_rank,
            pp_size=pp_size,
            pp_rank=0,
            save_dir=output_dir,  # Use the main output directory
            num_kv_heads=num_kv_heads,
            num_query_heads=num_query_heads,
            use_mla=use_mla,
        )
        
        print(f"Rank {tp_rank} completed: {rank_output_dir}")
    
    print(f"\nMulti-rank conversion completed!")
    print(f"Output directory: {output_dir}")
    
    # List the created files
    print("\nCreated files:")
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path) / (1024**3)  # GB
            print(f"  {file_path} ({file_size:.2f} GB)")

if __name__ == "__main__":
    # Test with MiniCPM4-8B
    create_multi_rank_megatron_checkpoint(
        hf_model_path="openbmb/MiniCPM4-8B",
        output_dir="./test_output_minicpm4_8b_megatron_multi_rank",
        num_layer=32,
        tp_size=2,
        pp_size=1,
        num_kv_heads=8,
        num_query_heads=32,
        use_mla=False,
    ) 