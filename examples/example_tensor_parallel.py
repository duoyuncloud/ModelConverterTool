#!/usr/bin/env python3
"""
Example script demonstrating tensor parallel and pipeline parallel conversion
Shows how to use the new converters for MiniCPM and Llama models
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from megatron_converters import (
    # Core converters
    TensorParallelConverter,
    SmartConverter,
    # TP/PP converters
    convert_8b_minicpm_megatron_to_hf,
    convert_3b_minicpm_megatron_to_hf,
    convert_7b_llama_megatron_to_hf,
    # Smart converters
    smart_convert_megatron_to_hf,
    smart_convert_hf_to_megatron,
    convert_minicpm_8b,
    convert_minicpm_3b,
    convert_llama_7b,
    # Direct checkpoint converters
    ckpt_to_hf_minicpm_tp_pp,
)


def example_basic_tp_conversion():
    """Example of basic tensor parallel conversion"""
    print("=== Basic Tensor Parallel Conversion Example ===")

    # Example paths (replace with actual paths)
    input_dir = "/path/to/megatron/checkpoint"
    output_path = "/path/to/output/hf_weights.pt"

    # Method 1: Using the class-based converter
    converter = TensorParallelConverter()
    converter.convert_minicpm_megatron_to_hf_tp_pp(
        num_layer=32, tp_size=2, pp_size=1, in_dir=input_dir, save_path=output_path, num_kv_heads=8, num_query_heads=32
    )

    # Method 2: Using convenience functions
    convert_8b_minicpm_megatron_to_hf(input_dir, output_path)
    convert_3b_minicpm_megatron_to_hf(input_dir, output_path)
    convert_7b_llama_megatron_to_hf(input_dir, output_path)


def example_smart_conversion():
    """Example of smart conversion with auto-detection"""
    print("=== Smart Conversion Example ===")

    # Example paths (replace with actual paths)
    input_dir = "/path/to/megatron/checkpoint"
    output_path = "/path/to/output/hf_weights.pt"

    # Method 1: One-click smart conversion
    smart_convert_megatron_to_hf(input_dir, output_path)

    # Method 2: Using the smart converter class
    converter = SmartConverter()
    converter.convert_megatron_to_hf(input_dir, output_path)

    # Method 3: Specify model type and size
    converter.convert_megatron_to_hf(input_dir, output_path, model_type="minicpm", model_size="8b")

    # Method 4: Model-specific convenience functions
    convert_minicpm_8b(input_dir, output_path)
    convert_minicpm_3b(input_dir, output_path)
    convert_llama_7b(input_dir, output_path)


def example_direct_checkpoint_conversion():
    """Example of direct checkpoint conversion"""
    print("=== Direct Checkpoint Conversion Example ===")

    # Example paths (replace with actual paths)
    input_dir = "/path/to/megatron/checkpoint"
    output_path = "/path/to/output/hf_weights.pt"

    # Using the direct checkpoint converter
    ckpt_to_hf_minicpm_tp_pp(
        num_layer=80, tp_size=8, pp_size=4, in_dir=input_dir, save_path=output_path, num_kv_heads=8, num_query_heads=64
    )


def example_hf_to_megatron_conversion():
    """Example of HF to Megatron conversion"""
    print("=== HF to Megatron Conversion Example ===")

    # Example paths (replace with actual paths)
    input_path = "/path/to/hf/weights.pt"
    output_dir = "/path/to/output/megatron/checkpoint"

    # Smart conversion HF to Megatron
    smart_convert_hf_to_megatron(input_path, output_dir)

    # Using the smart converter class
    converter = SmartConverter()
    converter.convert_hf_to_megatron(input_path, output_dir, model_type="minicpm", model_size="8b")


def example_advanced_configuration():
    """Example of advanced configuration"""
    print("=== Advanced Configuration Example ===")

    # Example paths (replace with actual paths)
    input_dir = "/path/to/megatron/checkpoint"
    output_path = "/path/to/output/hf_weights.pt"

    # Advanced configuration with custom parameters
    converter = TensorParallelConverter()

    # Custom model configuration
    model_config = {"num_layer": 40, "tp_size": 4, "pp_size": 2, "num_kv_heads": 8, "num_query_heads": 32}

    converter.convert_minicpm_megatron_to_hf_tp_pp(in_dir=input_dir, save_path=output_path, **model_config)


def example_error_handling():
    """Example of error handling"""
    print("=== Error Handling Example ===")

    try:
        # Try to convert with invalid parameters
        converter = TensorParallelConverter()
        converter.convert_minicpm_megatron_to_hf_tp_pp(
            num_layer=32,
            tp_size=3,  # This will cause an error (not divisible by num_kv_heads=8)
            pp_size=1,
            in_dir="/nonexistent/path",
            save_path="/nonexistent/output.pt",
            num_kv_heads=8,
            num_query_heads=32,
        )
    except AssertionError as e:
        print(f"Validation error: {e}")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def main():
    """Main function to run all examples"""
    print("Tensor Parallel Conversion Examples")
    print("=" * 50)

    # Note: These examples use placeholder paths
    # Replace with actual paths to run the examples

    print("\nNote: These examples use placeholder paths.")
    print("Replace the paths with actual checkpoint directories to run the examples.")
    print("=" * 50)

    # Run examples (commented out to avoid errors with placeholder paths)
    # example_basic_tp_conversion()
    # example_smart_conversion()
    # example_direct_checkpoint_conversion()
    # example_hf_to_megatron_conversion()
    # example_advanced_configuration()
    # example_error_handling()

    print("\nExample functions defined. Uncomment the function calls above to run them.")
    print("Make sure to replace placeholder paths with actual checkpoint directories.")


if __name__ == "__main__":
    main()
