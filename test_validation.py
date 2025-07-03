#!/usr/bin/env python3
"""
Test the updated validation logic with gptqmodel API
"""

import os
import sys
from pathlib import Path

# Set environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["MPS_VISIBLE_DEVICES"] = ""
os.environ["TRANSFORMERS_NO_MPS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from model_converter_tool.converter import ModelConverter
from model_converter_tool.validator import ModelValidator

def test_quantization_with_validation():
    """Test GPTQ quantization with the updated validation"""
    print("=== Testing GPTQ Quantization with Updated Validation ===")
    
    converter = ModelConverter()
    validator = ModelValidator()
    
    # Test quantization with validation enabled
    result = converter.convert(
        input_source="opt-125m-local",
        output_format="gptq",
        output_path="outputs/opt_125m_gptq_validated",
        model_type="text-generation",
        device="auto",
        validate=True,  # Enable validation with updated logic
    )
    
    print(f"Conversion result: {result}")
    
    if result["success"]:
        print("✅ GPTQ quantization completed successfully!")
        print(f"Output path: {result.get('output_path')}")
        
        # Check validation result
        model_validation = result.get("model_validation", {})
        if model_validation.get("success"):
            print("✅ Model validation passed!")
            print(f"Validation message: {model_validation.get('message')}")
            if model_validation.get("input_text"):
                print(f"Test input: '{model_validation.get('input_text')}'")
                print(f"Test output: '{model_validation.get('output_text')}'")
        else:
            print(f"⚠️ Model validation failed: {model_validation.get('error')}")
        
        # Check if output files exist
        output_path = Path(result.get('output_path', ''))
        if output_path.exists():
            files = list(output_path.glob("*"))
            print(f"Output files: {[f.name for f in files]}")
        else:
            print("⚠️ Output directory not found")
    else:
        print(f"❌ GPTQ quantization failed: {result.get('error')}")
    
    return result["success"]

def test_standalone_validation():
    """Test standalone validation of an existing quantized model"""
    print("\n=== Testing Standalone Validation ===")
    
    validator = ModelValidator()
    
    # Test validation on the previously created model
    model_path = "outputs/opt_125m_gptq_simple"
    
    if Path(model_path).exists():
        validation_result = validator.validate_converted_model(
            model_path=model_path,
            output_format="gptq",
            model_type="text-generation",
            quantization_type="gptq"
        )
        
        print(f"Standalone validation result: {validation_result}")
        
        if validation_result.get("success"):
            print("✅ Standalone validation passed!")
            print(f"Message: {validation_result.get('message')}")
        else:
            print(f"❌ Standalone validation failed: {validation_result.get('error')}")
    else:
        print(f"⚠️ Model path not found: {model_path}")

if __name__ == "__main__":
    success1 = test_quantization_with_validation()
    test_standalone_validation()
    sys.exit(0 if success1 else 1) 