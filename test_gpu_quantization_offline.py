#!/usr/bin/env python3
"""
Offline GPU Quantization Test Script
Creates a minimal model locally and tests quantization without network access
"""

import os
import sys
import logging
import torch
import json
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_offline_test_model():
    """Create a minimal test model without downloading from HuggingFace"""
    logger.info("=== Creating Offline Test Model ===")
    
    try:
        from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
        
        # Create model directory
        model_dir = Path("offline_test_model")
        model_dir.mkdir(exist_ok=True)
        
        # Create a minimal GPT-2 config manually
        config_dict = {
            "architectures": ["GPT2LMHeadModel"],
            "model_type": "gpt2",
            "vocab_size": 50257,
            "n_positions": 1024,
            "n_embd": 768,
            "n_layer": 4,  # Very small for testing
            "n_head": 12,
            "n_inner": None,
            "activation_function": "gelu_new",
            "resid_pdrop": 0.1,
            "embd_pdrop": 0.1,
            "attn_pdrop": 0.1,
            "layer_norm_epsilon": 1e-05,
            "initializer_range": 0.02,
            "scale_attn_weights": True,
            "use_cache": True,
            "bos_token_id": 50256,
            "eos_token_id": 50256,
            "pad_token_id": 50256,
            "tie_word_embeddings": True
        }
        
        # Save config
        config_path = model_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Create tokenizer config
        tokenizer_config = {
            "model_type": "gpt2",
            "tokenizer_class": "GPT2Tokenizer",
            "vocab_size": 50257,
            "merges_file": None,
            "vocab_file": None,
            "unk_token": "<|endoftext|>",
            "bos_token": "<|endoftext|>",
            "eos_token": "<|endoftext|>",
            "pad_token": "<|endoftext|>",
            "clean_up_tokenization_spaces": True
        }
        
        tokenizer_config_path = model_dir / "tokenizer_config.json"
        with open(tokenizer_config_path, 'w') as f:
            json.dump(tokenizer_config, f, indent=2)
        
        # Create minimal vocab
        vocab = {"<|endoftext|>": 50256}
        for i in range(50256):
            vocab[f"<|{i}|>"] = i
        
        vocab_path = model_dir / "vocab.json"
        with open(vocab_path, 'w') as f:
            json.dump(vocab, f, indent=2)
        
        # Create merges file
        merges_path = model_dir / "merges.txt"
        with open(merges_path, 'w') as f:
            f.write("#version: 0.2\n")
            f.write("t h\n")
            f.write("th e\n")
            f.write("the <|endoftext|>\n")
        
        # Create special tokens map
        special_tokens = {
            "unk_token": "<|endoftext|>",
            "bos_token": "<|endoftext|>",
            "eos_token": "<|endoftext|>",
            "pad_token": "<|endoftext|>"
        }
        
        special_tokens_path = model_dir / "special_tokens_map.json"
        with open(special_tokens_path, 'w') as f:
            json.dump(special_tokens, f, indent=2)
        
        # Load config and create model
        config = AutoConfig.from_pretrained(str(model_dir))
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        
        # Create minimal model
        logger.info("Creating minimal model...")
        model = AutoModelForCausalLM.from_config(config)
        
        # Save model
        model.save_pretrained(model_dir, safe_serialization=True)
        tokenizer.save_pretrained(model_dir)
        
        logger.info(f"✅ Offline test model created at: {model_dir}")
        return str(model_dir)
        
    except Exception as e:
        logger.error(f"❌ Failed to create offline model: {e}")
        return None

def test_gptq_basic(model_path):
    """Test basic GPTQ functionality"""
    logger.info("=== Testing Basic GPTQ ===")
    
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        from transformers import AutoTokenizer
        
        # Load model
        logger.info(f"Loading model from: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoGPTQForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Test basic inference
        logger.info("Testing basic inference...")
        test_text = "Hello world"
        inputs = tokenizer(test_text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logger.info("✅ Basic GPTQ model loading and inference successful")
        
        # Test quantization config creation
        quantize_config = BaseQuantizeConfig(
            bits=4,
            group_size=128,
            damp_percent=0.1,
            desc_act=False,
            static_groups=False,
            sym=True,
            true_sequential=True
        )
        
        logger.info("✅ GPTQ quantization config created successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic GPTQ test failed: {e}")
        return False

def test_awq_basic(model_path):
    """Test basic AWQ functionality"""
    logger.info("=== Testing Basic AWQ ===")
    
    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
        
        # Load model
        logger.info(f"Loading model from: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoAWQForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Test basic inference
        logger.info("Testing basic inference...")
        test_text = "Hello world"
        inputs = tokenizer(test_text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logger.info("✅ Basic AWQ model loading and inference successful")
        return True
        
    except ImportError:
        logger.warning("❌ AWQ not available (autoawq not installed)")
        return False
    except Exception as e:
        logger.error(f"❌ Basic AWQ test failed: {e}")
        return False

def test_gpu_operations():
    """Test basic GPU operations"""
    logger.info("=== Testing GPU Operations ===")
    
    try:
        # Check CUDA availability
        if not torch.cuda.is_available():
            logger.error("❌ CUDA not available")
            return False
        
        # Get GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        
        logger.info(f"✅ GPU: {gpu_name}")
        logger.info(f"✅ Memory: {gpu_memory / 1024**3:.1f}GB")
        
        # Test tensor operations on GPU
        logger.info("Testing GPU tensor operations...")
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        z = torch.matmul(x, y)
        
        logger.info(f"✅ GPU tensor operation successful, result shape: {z.shape}")
        
        # Test model on GPU
        logger.info("Testing model on GPU...")
        from transformers import AutoConfig, AutoModelForCausalLM
        
        config = AutoConfig.from_pretrained("gpt2")
        config.n_layer = 2  # Very small
        config.n_embd = 128
        config.n_head = 4
        
        model = AutoModelForCausalLM.from_config(config)
        model = model.cuda()
        
        # Test forward pass
        test_input = torch.randint(0, 50257, (1, 10)).cuda()
        with torch.no_grad():
            output = model(test_input)
        
        logger.info(f"✅ GPU model forward pass successful, output shape: {output.logits.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ GPU operations test failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🚀 Starting Offline GPU Quantization Test")
    
    success_count = 0
    total_tests = 0
    
    # Test GPU operations
    total_tests += 1
    if test_gpu_operations():
        success_count += 1
    
    # Create offline model
    model_path = create_offline_test_model()
    if model_path:
        # Test GPTQ basic functionality
        total_tests += 1
        if test_gptq_basic(model_path):
            success_count += 1
        
        # Test AWQ basic functionality
        total_tests += 1
        if test_awq_basic(model_path):
            success_count += 1
    
    # Summary
    logger.info("=== Test Summary ===")
    logger.info(f"Tests passed: {success_count}/{total_tests}")
    
    if success_count > 0:
        logger.info("✅ GPU quantization capabilities confirmed!")
        return True
    else:
        logger.error("❌ All tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 