#!/usr/bin/env python3
"""
Test script to verify GGUF conversion functionality
"""

import os
from pathlib import Path
from model_converter_tool.converter import ModelConverter

def test_gguf_conversion():
    """Test GGUF conversion with detailed output"""
    print("🔍 Testing GGUF Conversion...")
    
    converter = ModelConverter()
    
    # Use outputs directory instead of temporary directory
    output_path = Path("outputs") / "gguf_test"
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output directory: {output_path}")
    
    # Test GGUF conversion
    result = converter.convert(
        input_source="hf:distilbert-base-uncased",
        output_format="gguf",
        output_path=str(output_path),
        model_type="text-classification",
        quantization="q4_0",
    )
    
    print(f"✅ Conversion result: {result}")
    
    # Check if files were created
    if result.get("success", False):
        print("📋 Generated files:")
        for file_path in output_path.rglob("*"):
            if file_path.is_file():
                size = file_path.stat().st_size
                print(f"  - {file_path.name}: {size} bytes")
                
                # Check if it's a GGUF file
                if file_path.suffix == ".gguf":
                    print(f"    🔍 Checking GGUF file: {file_path}")
                    with open(file_path, 'rb') as f:
                        header = f.read(8)
                        if header.startswith(b'GGUF'):
                            print("    ✅ Valid GGUF file header detected!")
                        else:
                            print(f"    ⚠️  Unexpected header: {header[:4]}")
    else:
        print(f"❌ Conversion failed: {result.get('error', 'Unknown error')}")
        
    return result

if __name__ == "__main__":
    test_gguf_conversion() 