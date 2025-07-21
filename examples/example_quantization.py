"""
Quantization example: Convert and quantize a model in one step.
"""

import os

# Simple quantization: convert and quantize a model to GGUF with q4 quantization
os.system("python -m model_converter_tool.cli convert gpt2 --to gguf --output outputs/gpt2-q4.gguf --quant q4")

# Advanced: use a quantization config file (YAML/JSON) for more options
# os.system(
#     "python -m model_converter_tool.cli convert gpt2 --to gptq --output outputs/gpt2-gptq --quant-config ../configs/quant_config.yaml"
# )
