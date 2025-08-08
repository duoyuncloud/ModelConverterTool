#!/usr/bin/env python3
"""
Test reverse conversion with a minimal multi-rank checkpoint
"""

import torch
import os
from megatron_converters import convert_minicpm_megatron_to_hf_tp_pp

def create_minimal_multi_rank_checkpoint():
    """Create a minimal multi-rank checkpoint for testing"""
    base_dir = "./test_minimal_multi_rank"
    os.makedirs(base_dir, exist_ok=True)
    
    # Create rank 0
    rank0_dir = os.path.join(base_dir, "mp_rank_00")
    os.makedirs(rank0_dir, exist_ok=True)
    
    # Create rank 1
    rank1_dir = os.path.join(base_dir, "mp_rank_01")
    os.makedirs(rank1_dir, exist_ok=True)
    
    # Create minimal tensors for rank 0
    rank0_state_dict = {
        "embedding.word_embeddings.weight": torch.randn(1000, 512),  # Half vocab size
        "decoder.final_layernorm.weight": torch.randn(512),
        "output_layer.weight": torch.randn(1000, 512),  # Half vocab size
        "decoder.layers.0.self_attention.linear_qkv.weight": torch.randn(768, 512),  # Half attention size
        "decoder.layers.0.self_attention.linear_qkv.bias": torch.randn(768),  # Add bias
        "decoder.layers.0.self_attention.linear_proj.weight": torch.randn(512, 384),  # Half output size
        "decoder.layers.0.self_attention.linear_proj.bias": torch.randn(512),  # Add bias
        "decoder.layers.0.self_attention.linear_qkv.layer_norm_weight": torch.randn(512),
        "decoder.layers.0.mlp.linear_fc1.weight": torch.randn(1024, 512),  # Half MLP size
        "decoder.layers.0.mlp.linear_fc1.bias": torch.randn(1024),  # Add bias
        "decoder.layers.0.mlp.linear_fc2.weight": torch.randn(512, 1024),  # Half MLP size
        "decoder.layers.0.mlp.linear_fc2.bias": torch.randn(512),  # Add bias
        "decoder.layers.0.mlp.linear_fc1.layer_norm_weight": torch.randn(512),
    }
    
    # Create minimal tensors for rank 1
    rank1_state_dict = {
        "embedding.word_embeddings.weight": torch.randn(1000, 512),  # Other half vocab size
        "decoder.final_layernorm.weight": torch.randn(512),
        "output_layer.weight": torch.randn(1000, 512),  # Other half vocab size
        "decoder.layers.0.self_attention.linear_qkv.weight": torch.randn(768, 512),  # Other half attention size
        "decoder.layers.0.self_attention.linear_qkv.bias": torch.randn(768),  # Add bias
        "decoder.layers.0.self_attention.linear_proj.weight": torch.randn(512, 384),  # Other half output size
        "decoder.layers.0.self_attention.linear_proj.bias": torch.randn(512),  # Add bias
        "decoder.layers.0.self_attention.linear_qkv.layer_norm_weight": torch.randn(512),
        "decoder.layers.0.mlp.linear_fc1.weight": torch.randn(1024, 512),  # Other half MLP size
        "decoder.layers.0.mlp.linear_fc1.bias": torch.randn(1024),  # Add bias
        "decoder.layers.0.mlp.linear_fc2.weight": torch.randn(512, 1024),  # Other half MLP size
        "decoder.layers.0.mlp.linear_fc2.bias": torch.randn(512),  # Add bias
        "decoder.layers.0.mlp.linear_fc1.layer_norm_weight": torch.randn(512),
    }
    
    # Save checkpoints
    torch.save({"model": rank0_state_dict}, os.path.join(rank0_dir, "model_optim_rng.pt"))
    torch.save({"model": rank1_state_dict}, os.path.join(rank1_dir, "model_optim_rng.pt"))
    
    print(f"Created minimal multi-rank checkpoint in {base_dir}")
    return base_dir

def test_reverse_conversion():
    """Test reverse conversion with minimal checkpoint"""
    print("Creating minimal multi-rank checkpoint...")
    checkpoint_dir = create_minimal_multi_rank_checkpoint()
    
    print("Testing reverse conversion...")
    try:
        convert_minicpm_megatron_to_hf_tp_pp(
            num_layer=1,  # Just one layer for testing
            tp_size=2,
            pp_size=1,
            in_dir=checkpoint_dir,
            save_path="./test_minimal_hf_reverse",
            num_kv_heads=2,
            num_query_heads=8
        )
        print("✅ Reverse conversion successful!")
        
        # Check output
        if os.path.exists("./test_minimal_hf_reverse"):
            print("✅ Output directory created successfully!")
            files = os.listdir("./test_minimal_hf_reverse")
            print(f"Output files: {files}")
        else:
            print("❌ Output directory not created")
            
    except Exception as e:
        print(f"❌ Reverse conversion failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    import shutil
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    shutil.rmtree("./test_minimal_hf_reverse", ignore_errors=True)

if __name__ == "__main__":
    test_reverse_conversion() 