"""
Basic example: Convert a model from one format to another using the CLI entry point.
"""
import os

# Example command: convert a HuggingFace model to GGUF format
os.system(
    "python -m model_converter_tool.cli convert path/to/input_model --to gguf --output path/to/output_model.gguf"
) 