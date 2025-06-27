#!/usr/bin/env python3
"""
Demo script for GPT-2 model conversion with validation
"""

import tempfile
import os
from pathlib import Path
from model_converter_tool.converter import ModelConverter
from model_converter_tool.validator import ModelValidator

def demo_gpt2_conversion():
    """Demonstrate GPT-2 conversion with validation"""
    print("🚀 GPT-2 Model Conversion Demo")
    print("=" * 50)
    
    converter = ModelConverter()
    validator = ModelValidator()
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test different formats
        formats_to_test = [
            ("onnx", "text-generation", "simplify"),
            ("fp16", "text-generation", "prune"),
            ("torchscript", "text-generation", "optimize"),
            ("gguf", "text-generation", None),
        ]
        
        results = []
        
        for output_format, model_type, postprocess in formats_to_test:
            print(f"\n📋 Testing {output_format.upper()} conversion...")
            
            # Create output directory
            out_dir = temp_path / f"gpt2_{output_format}"
            out_dir.mkdir()
            
            # Convert model
            result = converter.convert(
                input_source="hf:gpt2",
                output_format=output_format,
                output_path=str(out_dir),
                model_type=model_type,
                postprocess=postprocess,
                validate=True,  # Enable validation
            )
            
            # Display results
            if result["success"]:
                print(f"✅ {output_format.upper()} conversion successful")
                print(f"   Output: {result.get('output_path', 'N/A')}")
                print(f"   Validation: {result.get('validation', 'N/A')}")
                
                # Show model validation results
                model_validation = result.get('model_validation')
                if model_validation and model_validation.get('success'):
                    print(f"   Model validation: ✅ {model_validation.get('message', 'Passed')}")
                elif model_validation:
                    print(f"   Model validation: ⚠️  {model_validation.get('error', 'Failed')}")
                else:
                    print(f"   Model validation: ⏭️  Skipped")
                
                # Show postprocess results
                postprocess_result = result.get('postprocess_result')
                if postprocess_result:
                    print(f"   Postprocess: {postprocess_result}")
                
                results.append({
                    "format": output_format,
                    "success": True,
                    "validation": result.get('validation', False),
                    "model_validation": model_validation.get('success', False) if model_validation else False
                })
            else:
                print(f"❌ {output_format.upper()} conversion failed")
                print(f"   Error: {result.get('error', 'Unknown error')}")
                results.append({
                    "format": output_format,
                    "success": False,
                    "validation": False,
                    "model_validation": False
                })
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 Conversion Summary")
        print("=" * 50)
        
        successful_conversions = [r for r in results if r["success"]]
        successful_validations = [r for r in results if r["success"] and r["validation"]]
        successful_model_validations = [r for r in results if r["success"] and r["model_validation"]]
        
        print(f"Total formats tested: {len(results)}")
        print(f"Successful conversions: {len(successful_conversions)}/{len(results)}")
        print(f"Successful validations: {len(successful_validations)}/{len(results)}")
        print(f"Successful model validations: {len(successful_model_validations)}/{len(results)}")
        
        print("\n📋 Detailed Results:")
        for result in results:
            status = "✅" if result["success"] else "❌"
            validation_status = "✅" if result["validation"] else "❌"
            model_validation_status = "✅" if result["model_validation"] else "❌"
            
            print(f"  {result['format'].upper():12} | Conversion: {status} | Validation: {validation_status} | Model Validation: {model_validation_status}")
        
        return results

def demo_batch_conversion():
    """Demonstrate batch conversion with validation"""
    print("\n🔄 Batch Conversion Demo")
    print("=" * 50)
    
    converter = ModelConverter()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create batch tasks
        tasks = [
            {
                "input_source": "hf:gpt2",
                "output_format": "onnx",
                "output_path": str(temp_path / "batch_onnx"),
                "model_type": "text-generation",
                "postprocess": "simplify",
                "validate": True,
            },
            {
                "input_source": "hf:gpt2",
                "output_format": "fp16",
                "output_path": str(temp_path / "batch_fp16"),
                "model_type": "text-generation",
                "postprocess": "prune",
                "validate": True,
            },
            {
                "input_source": "hf:gpt2",
                "output_format": "torchscript",
                "output_path": str(temp_path / "batch_torchscript"),
                "model_type": "text-generation",
                "postprocess": "optimize",
                "validate": True,
            },
        ]
        
        print(f"Running batch conversion with {len(tasks)} tasks...")
        results = converter.batch_convert(tasks, max_workers=1)
        
        print("\n📊 Batch Conversion Results:")
        for i, result in enumerate(results):
            task = tasks[i]
            status = "✅" if result["success"] else "❌"
            validation_status = "✅" if result.get("validation", False) else "❌"
            
            # Fix the model validation check
            model_validation = result.get("model_validation")
            if model_validation and isinstance(model_validation, dict):
                model_validation_status = "✅" if model_validation.get("success", False) else "❌"
            else:
                model_validation_status = "❌"
            
            print(f"  Task {i+1}: {task['output_format'].upper():12} | Conversion: {status} | Validation: {validation_status} | Model Validation: {model_validation_status}")
        
        successful = sum(1 for r in results if r["success"])
        print(f"\n✅ Batch conversion completed: {successful}/{len(results)} successful")

if __name__ == "__main__":
    try:
        # Run individual conversion demo
        demo_gpt2_conversion()
        
        # Run batch conversion demo
        demo_batch_conversion()
        
        print("\n🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc() 