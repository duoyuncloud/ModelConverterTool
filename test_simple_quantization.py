#!/usr/bin/env python3
"""
Simple GPTQ quantization test without validation
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

def test_gptq_quantization():
    """Test GPTQ quantization without validation"""
    print("=== Testing GPTQ Quantization (No Validation) ===")
    
    converter = ModelConverter()
    
    result = converter.convert(
        input_source="opt-125m-local",
        output_format="gptq",
        output_path="outputs/opt_125m_gptq_simple",
        model_type="text-generation",
        device="auto",
        validate=False,  # Disable validation to avoid segfault
    )
    
    print(f"Conversion result: {result}")
    
    if result["success"]:
        print("✅ GPTQ quantization completed successfully!")
        print(f"Output path: {result.get('output_path')}")
        
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

if __name__ == "__main__":
    success = test_gptq_quantization()
    sys.exit(0 if success else 1) 