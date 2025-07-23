"""
Basic example: Convert a model from one format to another using the CLI.
"""

import os

# Convert a HuggingFace model to GGUF format
os.system("python -m model_converter_tool.cli convert gpt2 --to gguf --output outputs/gpt2.gguf")
