"""
Quantization example: Convert and quantize a model in one step.
"""
import os

# Example command: convert and quantize a model to GGUF with q4 quantization
os.system(
    "python -m model_converter_tool.cli convert path/to/input_model --to gguf --output path/to/output_model-q4.gguf --quant q4"
) 