#!/usr/bin/env python3
"""
Comprehensive GPU Quantization Test Script
Tests real GPTQ/AWQ quantization conversion on GPU server
"""

import os
import sys
import logging
import tempfile
import shutil
from pathlib import Path
import torch
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('gpu_quantization_test.log')
    ]
)
logger = logging.getLogger(__name__)

def check_gpu_availability():
    """Check GPU availability and CUDA support"""
    logger.info("=== GPU Availability Check ===")
    
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available")
        return False
    
    gpu_count = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    cuda_version = torch.version.cuda
    
    logger.info(f"✅ CUDA available: {cuda_version}")
    logger.info(f"✅ GPU count: {gpu_count}")
    logger.info(f"✅ GPU name: {gpu_name}")
    
    # Test GPU memory
    try:
        torch.cuda.empty_cache()
        memory_allocated = torch.cuda.memory_allocated(0)
        memory_reserved = torch.cuda.memory_reserved(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory
        
        logger.info(f"✅ GPU memory: {total_memory / 1024**3:.1f}GB total")
        logger.info(f"✅ Memory allocated: {memory_allocated / 1024**2:.1f}MB")
        logger.info(f"✅ Memory reserved: {memory_reserved / 1024**2:.1f}MB")
        
        return True
    except Exception as e:
        logger.error(f"❌ GPU memory test failed: {e}")
        return False

def check_dependencies():
    """Check required dependencies for quantization"""
    logger.info("=== Dependency Check ===")
    
    dependencies = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'auto_gptq': 'AutoGPTQ',
        'gptqmodel': 'GPTQModel',
        'safetensors': 'Safetensors'
    }
    
    missing_deps = []
    available_deps = {}
    
    for module, name in dependencies.items():
        try:
            if module == 'torch':
                import torch
                version = torch.__version__
                cuda_available = torch.cuda.is_available()
                logger.info(f"✅ {name}: {version} (CUDA: {cuda_available})")
            elif module == 'transformers':
                import transformers
                version = transformers.__version__
                logger.info(f"✅ {name}: {version}")
            elif module == 'auto_gptq':
                import auto_gptq
                version = auto_gptq.__version__
                logger.info(f"✅ {name}: {version}")
            elif module == 'gptqmodel':
                import gptqmodel
                version = gptqmodel.__version__
                logger.info(f"✅ {name}: {version}")
            elif module == 'safetensors':
                import safetensors
                version = safetensors.__version__
                logger.info(f"✅ {name}: {version}")
            
            available_deps[module] = version
            
        except ImportError:
            logger.warning(f"❌ {name} not found")
            missing_deps.append(module)
    
    if missing_deps:
        logger.warning(f"Missing dependencies: {missing_deps}")
        return False
    
    return True

def create_minimal_test_model():
    """Create a minimal test model for quantization"""
    logger.info("=== Creating Minimal Test Model ===")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
        
        # Create a minimal GPT-2 like config
        config = AutoConfig.from_pretrained("gpt2", trust_remote_code=True)
        config.vocab_size = 50257
        config.n_positions = 1024
        config.n_embd = 768
        config.n_layer = 6  # Smaller model for testing
        config.n_head = 12
        config.model_type = "gpt2"
        
        # Create model directory
        model_dir = Path("test_minimal_model")
        model_dir.mkdir(exist_ok=True)
        
        # Save config
        config.save_pretrained(model_dir)
        
        # Create and save tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2", trust_remote_code=True)
        tokenizer.save_pretrained(model_dir)
        
        # Create minimal model
        logger.info("Creating minimal model...")
        model = AutoModelForCausalLM.from_config(config)
        
        # Save model
        model.save_pretrained(model_dir, safe_serialization=True)
        
        logger.info(f"✅ Minimal test model created at: {model_dir}")
        return str(model_dir)
        
    except Exception as e:
        logger.error(f"❌ Failed to create minimal model: {e}")
        return None

def test_gptq_quantization(model_path):
    """Test GPTQ quantization"""
    logger.info("=== Testing GPTQ Quantization ===")
    
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        from transformers import AutoTokenizer
        
        # Load model and tokenizer
        logger.info(f"Loading model from: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoGPTQForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Create quantization config
        quantize_config = BaseQuantizeConfig(
            bits=4,
            group_size=128,
            damp_percent=0.1,
            desc_act=False,
            static_groups=False,
            sym=True,
            true_sequential=True,
            model_name_or_path=None,
            model_file_base_name="model"
        )
        
        # Prepare calibration data
        logger.info("Preparing calibration data...")
        calibration_data = []
        for i in range(10):  # Small calibration set
            text = f"This is calibration example {i} for GPTQ quantization testing."
            tokens = tokenizer(text, return_tensors="pt")
            calibration_data.append(tokens)
        
        # Quantize model
        logger.info("Starting GPTQ quantization...")
        quantized_model = AutoGPTQForCausalLM.quantize_model(
            model,
            quantize_config,
            calibration_data
        )
        
        # Save quantized model
        output_dir = Path("gptq_quantized_output")
        output_dir.mkdir(exist_ok=True)
        
        quantized_model.save_quantized(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        logger.info(f"✅ GPTQ quantization successful! Output: {output_dir}")
        return True
        
    except Exception as e:
        logger.error(f"❌ GPTQ quantization failed: {e}")
        return False

def test_awq_quantization(model_path):
    """Test AWQ quantization"""
    logger.info("=== Testing AWQ Quantization ===")
    
    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
        
        # Load model and tokenizer
        logger.info(f"Loading model from: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoAWQForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Prepare calibration data
        logger.info("Preparing calibration data...")
        calibration_data = []
        for i in range(10):  # Small calibration set
            text = f"This is calibration example {i} for AWQ quantization testing."
            tokens = tokenizer(text, return_tensors="pt")
            calibration_data.append(tokens)
        
        # Quantize model
        logger.info("Starting AWQ quantization...")
        quantized_model = AutoAWQForCausalLM.quantize(
            model,
            calibration_data,
            bits=4,
            group_size=128
        )
        
        # Save quantized model
        output_dir = Path("awq_quantized_output")
        output_dir.mkdir(exist_ok=True)
        
        quantized_model.save_quantized(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        logger.info(f"✅ AWQ quantization successful! Output: {output_dir}")
        return True
        
    except ImportError:
        logger.warning("❌ AWQ not available (autoawq not installed)")
        return False
    except Exception as e:
        logger.error(f"❌ AWQ quantization failed: {e}")
        return False

def test_model_converter_tool(model_path):
    """Test using the ModelConverterTool"""
    logger.info("=== Testing ModelConverterTool ===")
    
    try:
        # Import the converter tool
        sys.path.insert(0, str(Path(__file__).parent))
        from model_converter_tool.converter import ModelConverter
        
        # Test GPTQ conversion
        logger.info("Testing GPTQ conversion with ModelConverterTool...")
        converter = ModelConverter()
        
        # GPTQ test
        gptq_result = converter.convert(
            model_path=model_path,
            output_format="gptq",
            output_path="gptq_converter_output",
            quantization_config={
                "bits": 4,
                "group_size": 128,
                "damp_percent": 0.1
            }
        )
        
        if gptq_result:
            logger.info("✅ ModelConverterTool GPTQ conversion successful!")
        else:
            logger.warning("⚠️ ModelConverterTool GPTQ conversion failed")
        
        # AWQ test
        logger.info("Testing AWQ conversion with ModelConverterTool...")
        awq_result = converter.convert(
            model_path=model_path,
            output_format="awq",
            output_path="awq_converter_output",
            quantization_config={
                "bits": 4,
                "group_size": 128
            }
        )
        
        if awq_result:
            logger.info("✅ ModelConverterTool AWQ conversion successful!")
        else:
            logger.warning("⚠️ ModelConverterTool AWQ conversion failed")
        
        return gptq_result or awq_result
        
    except Exception as e:
        logger.error(f"❌ ModelConverterTool test failed: {e}")
        return False

def cleanup_test_files():
    """Clean up test files"""
    logger.info("=== Cleaning up test files ===")
    
    test_dirs = [
        "test_minimal_model",
        "gptq_quantized_output", 
        "awq_quantized_output",
        "gptq_converter_output",
        "awq_converter_output"
    ]
    
    for dir_name in test_dirs:
        if Path(dir_name).exists():
            try:
                shutil.rmtree(dir_name)
                logger.info(f"✅ Cleaned up: {dir_name}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to clean up {dir_name}: {e}")

def main():
    """Main test function"""
    logger.info("🚀 Starting Comprehensive GPU Quantization Test")
    
    # Check GPU availability
    if not check_gpu_availability():
        logger.error("❌ GPU not available, exiting")
        return False
    
    # Check dependencies
    if not check_dependencies():
        logger.error("❌ Missing dependencies, exiting")
        return False
    
    # Create minimal test model
    model_path = create_minimal_test_model()
    if not model_path:
        logger.error("❌ Failed to create test model, exiting")
        return False
    
    success_count = 0
    total_tests = 0
    
    try:
        # Test GPTQ quantization
        total_tests += 1
        if test_gptq_quantization(model_path):
            success_count += 1
        
        # Test AWQ quantization
        total_tests += 1
        if test_awq_quantization(model_path):
            success_count += 1
        
        # Test ModelConverterTool
        total_tests += 1
        if test_model_converter_tool(model_path):
            success_count += 1
        
    finally:
        # Cleanup
        cleanup_test_files()
    
    # Summary
    logger.info("=== Test Summary ===")
    logger.info(f"Tests passed: {success_count}/{total_tests}")
    
    if success_count > 0:
        logger.info("✅ GPU quantization is working!")
        return True
    else:
        logger.error("❌ All quantization tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 