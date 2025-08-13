#!/usr/bin/env python3
"""
Comprehensive test for full bidirectional MiniCPM4 conversion
Tests both HF -> Megatron and Megatron -> HF with verification
"""

import os
import sys
import torch
import tempfile
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_dummy_minicpm4_model(temp_path):
    """Create a dummy MiniCPM4 model for testing"""
    hf_model_path = temp_path / "dummy_minicpm4_model"
    hf_model_path.mkdir(exist_ok=True)
    
    # Small model for quick testing
    vocab_size = 1000
    hidden_size = 512
    num_layers = 4
    num_heads = 8
    num_kv_heads = 2
    intermediate_size = 1024
    
    # Create dummy weights file
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
    
    # Save dummy weights
    torch.save(dummy_weights, hf_model_path / "pytorch_model.bin")
    
    # Create config
    config = {
        "architectures": ["MiniCPMForCausalLM"],
        "model_type": "minicpm",
        "hidden_size": hidden_size,
        "num_attention_heads": num_heads,
        "num_hidden_layers": num_layers,
        "intermediate_size": intermediate_size,
        "max_position_embeddings": 32768,
        "vocab_size": vocab_size,
        "rms_norm_eps": 1e-5,
    }
    
    import json
    with open(hf_model_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    return hf_model_path, {
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "intermediate_size": intermediate_size,
    }

def test_full_bidirectional_conversion():
    """Test complete bidirectional conversion cycle"""
    print("=== Testing Full Bidirectional Conversion ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create dummy model
        hf_original_path, model_config = create_dummy_minicpm4_model(temp_path)
        megatron_dir = temp_path / "megatron_output"
        hf_converted_dir = temp_path / "hf_converted"
        
        print(f"Model config: {model_config}")
        
        # Step 1: HF -> Megatron conversion
        print("\nStep 1: HF -> Megatron conversion...")
        try:
            from megatron_converters import convert_hf_to_megatron_minicpm4
            
            convert_hf_to_megatron_minicpm4(
                checkpoint_path=str(hf_original_path / "pytorch_model.bin"),
                output_path=str(megatron_dir),
                num_layer=model_config["num_layers"],
                tp_size=2,
                pp_size=1,
                num_kv_heads=model_config["num_kv_heads"],
                num_query_heads=model_config["num_heads"],
                dense_layer_ids="0,1,2,3",
                use_mla=False,
            )
            
            # Check output
            if megatron_dir.exists():
                files = list(megatron_dir.glob("mp_rank_*"))
                print(f"✓ HF -> Megatron successful! Generated {len(files)} files")
                hf_to_megatron_success = True
            else:
                print("✗ HF -> Megatron failed")
                hf_to_megatron_success = False
                
        except Exception as e:
            print(f"✗ HF -> Megatron failed: {e}")
            hf_to_megatron_success = False
        
        # Step 2: Megatron -> HF conversion
        print("\nStep 2: Megatron -> HF conversion...")
        try:
            from megatron_converters import convert_minicpm4_megatron_to_hf
            
            convert_minicpm4_megatron_to_hf(
                checkpoint_path=str(megatron_dir),
                output_path=str(hf_converted_dir),
                num_layer=model_config["num_layers"],
                tp_size=2,
                pp_size=1,
                num_kv_heads=model_config["num_kv_heads"],
                num_query_heads=model_config["num_heads"],
                hidden_size=model_config["hidden_size"],
                vocab_size=model_config["vocab_size"],
                intermediate_size=model_config["intermediate_size"],
            )
            
            # Check output
            if hf_converted_dir.exists():
                hf_files = list(hf_converted_dir.glob("*.bin")) + list(hf_converted_dir.glob("*.safetensors"))
                config_files = list(hf_converted_dir.glob("*.json"))
                print(f"✓ Megatron -> HF successful! Generated {len(hf_files)} weight files and {len(config_files)} config files")
                megatron_to_hf_success = True
            else:
                print("✗ Megatron -> HF failed")
                megatron_to_hf_success = False
                
        except Exception as e:
            print(f"✗ Megatron -> HF failed: {e}")
            import traceback
            traceback.print_exc()
            megatron_to_hf_success = False
        
        # Step 3: Verify conversion quality
        print("\nStep 3: Verifying conversion quality...")
        if hf_to_megatron_success and megatron_to_hf_success:
            try:
                # Load original and converted weights
                original_weights = torch.load(hf_original_path / "pytorch_model.bin", map_location="cpu")
                converted_weights = torch.load(hf_converted_dir / "pytorch_model.bin", map_location="cpu")
                
                # Check if all keys are present
                original_keys = set(original_weights.keys())
                converted_keys = set(converted_weights.keys())
                
                print(f"Original model has {len(original_keys)} parameters")
                print(f"Converted model has {len(converted_keys)} parameters")
                
                # Check key overlap
                missing_keys = original_keys - converted_keys
                extra_keys = converted_keys - original_keys
                
                if missing_keys:
                    print(f"⚠️ Missing keys in converted model: {missing_keys}")
                if extra_keys:
                    print(f"⚠️ Extra keys in converted model: {extra_keys}")
                
                # Check shapes for common keys
                common_keys = original_keys & converted_keys
                shape_matches = 0
                for key in common_keys:
                    if original_weights[key].shape == converted_weights[key].shape:
                        shape_matches += 1
                    else:
                        print(f"⚠️ Shape mismatch for {key}: {original_weights[key].shape} vs {converted_weights[key].shape}")
                
                print(f"✓ Shape matches: {shape_matches}/{len(common_keys)}")
                
                if len(missing_keys) == 0 and len(extra_keys) == 0 and shape_matches == len(common_keys):
                    print("✓ Conversion quality verification passed!")
                    verification_success = True
                else:
                    print("⚠️ Conversion quality verification has some issues")
                    verification_success = False
                    
            except Exception as e:
                print(f"✗ Verification failed: {e}")
                verification_success = False
        else:
            verification_success = False
        
        return hf_to_megatron_success, megatron_to_hf_success, verification_success

def test_imports():
    """Test that all functions can be imported correctly"""
    print("=== Testing Function Imports ===")
    
    try:
        from megatron_converters import (
            convert_hf_to_megatron_minicpm4,
            convert_minicpm4_megatron_to_hf,
        )
        
        print("✓ Successfully imported conversion functions")
        
        # Check if functions are callable
        if callable(convert_hf_to_megatron_minicpm4):
            print("✓ convert_hf_to_megatron_minicpm4 is callable")
        else:
            print("✗ convert_hf_to_megatron_minicpm4 is not callable")
        
        if callable(convert_minicpm4_megatron_to_hf):
            print("✓ convert_minicpm4_megatron_to_hf is callable")
        else:
            print("✗ convert_minicpm4_megatron_to_hf is not callable")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def main():
    """Main test function"""
    print("MiniCPM4-8B Full Bidirectional Conversion Test")
    print("=" * 60)
    
    # Test imports
    import_success = test_imports()
    
    # Test full bidirectional conversion
    hf_to_megatron_success, megatron_to_hf_success, verification_success = test_full_bidirectional_conversion()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Results:")
    print(f"Function Imports: {'✓ PASS' if import_success else '✗ FAIL'}")
    print(f"HF -> Megatron: {'✓ PASS' if hf_to_megatron_success else '✗ FAIL'}")
    print(f"Megatron -> HF: {'✓ PASS' if megatron_to_hf_success else '✗ FAIL'}")
    print(f"Quality Verification: {'✓ PASS' if verification_success else '✗ FAIL'}")
    
    if all([import_success, hf_to_megatron_success, megatron_to_hf_success, verification_success]):
        print("\n🎉 SUCCESS! Full bidirectional conversion is working!")
        print("MiniCPM4-8B can now be converted in both directions:")
        print("  - HuggingFace ↔ Megatron")
        print("  - Supports multi-rank distributed conversion")
        print("  - Maintains model integrity")
    else:
        print("\n⚠️ Some issues remain. Check the error messages above.")
    
    print("\n" + "=" * 60)
    print("Test completed!")

if __name__ == "__main__":
    main() 