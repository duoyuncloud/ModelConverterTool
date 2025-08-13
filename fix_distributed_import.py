#!/usr/bin/env python3
"""
Fix for distributed import issues in MiniCPM4 conversion
"""

import os
import sys
import tempfile
from pathlib import Path

def create_distributed_test_script():
    """Create a distributed test script with proper import paths"""
    
    script_content = '''#!/usr/bin/env python3
"""
Distributed test script for MiniCPM4 conversion with fixed imports
"""

import os
import sys
import torch
import tempfile
from pathlib import Path

# Fix import paths for distributed environment
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Now import the conversion function
from megatron_converters.hf_to_megatron_minicpm4 import convert_hf_to_megatron_minicpm4_main

def main():
    # Get rank from environment
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    print(f"Running on rank {rank}/{world_size}")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create dummy model
        hf_model_path = temp_path / "dummy_hf_model"
        hf_model_path.mkdir(exist_ok=True)
        
        # Create dummy weights (small model)
        vocab_size = 1000
        hidden_size = 512
        num_layers = 4
        num_heads = 8
        num_kv_heads = 2
        intermediate_size = 1024
        
        dummy_weights = {
            "model.embed_tokens.weight": torch.randn(vocab_size, hidden_size, dtype=torch.float32),
            "model.norm.weight": torch.randn(hidden_size, dtype=torch.float32),
        }
        
        # Add layer weights
        for layer_idx in range(num_layers):
            dummy_weights[f"model.layers.{layer_idx}.input_layernorm.weight"] = torch.randn(hidden_size, dtype=torch.float32)
            dummy_weights[f"model.layers.{layer_idx}.self_attn.q_proj.weight"] = torch.randn(num_heads * 64, hidden_size, dtype=torch.float32)
            dummy_weights[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = torch.randn(num_kv_heads * 64, hidden_size, dtype=torch.float32)
            dummy_weights[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = torch.randn(num_kv_heads * 64, hidden_size, dtype=torch.float32)
            dummy_weights[f"model.layers.{layer_idx}.self_attn.o_proj.weight"] = torch.randn(hidden_size, hidden_size, dtype=torch.float32)
            dummy_weights[f"model.layers.{layer_idx}.post_attention_layernorm.weight"] = torch.randn(hidden_size, dtype=torch.float32)
            dummy_weights[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = torch.randn(intermediate_size, hidden_size, dtype=torch.float32)
            dummy_weights[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = torch.randn(intermediate_size, hidden_size, dtype=torch.float32)
            dummy_weights[f"model.layers.{layer_idx}.mlp.down_proj.weight"] = torch.randn(hidden_size, intermediate_size, dtype=torch.float32)
        
        torch.save(dummy_weights, hf_model_path / "pytorch_model.bin")
        
        # Convert
        megatron_output = temp_path / "megatron_output"
        
        convert_hf_to_megatron_minicpm4_main(
            load_path=str(hf_model_path / "pytorch_model.bin"),
            num_layer=num_layers,
            tp_size=world_size,
            tp_rank=rank,
            pp_size=1,
            pp_rank=0,
            save_dir=str(megatron_output),
            num_kv_heads=num_kv_heads,
            num_query_heads=num_heads,
            dense_layer_ids="0,1,2,3",
            use_mla=False,
        )
        
        print(f"Rank {rank} conversion completed!")

if __name__ == "__main__":
    main()
'''
    
    # Save the script
    script_path = Path("distributed_test_fixed.py")
    with open(script_path, "w") as f:
        f.write(script_content)
    
    return script_path

def run_distributed_test():
    """Run the distributed test with fixed imports"""
    
    script_path = create_distributed_test_script()
    
    try:
        import subprocess
        
        print("Running distributed test with fixed imports...")
        result = subprocess.run([
            sys.executable, "-m", "torch.distributed.launch",
            "--nproc_per_node=2",
            str(script_path)
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✓ Distributed test successful!")
            print("Output:")
            print(result.stdout)
            return True
        else:
            print("✗ Distributed test failed!")
            print("Error:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"✗ Distributed test failed: {e}")
        return False
    finally:
        # Clean up
        if script_path.exists():
            script_path.unlink()

if __name__ == "__main__":
    print("Testing distributed conversion with fixed imports...")
    success = run_distributed_test()
    
    if success:
        print("\n✓ Distributed import issue fixed!")
    else:
        print("\n✗ Still need to investigate further") 