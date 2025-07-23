"""
muP-to-LLaMA example: Convert a muP-initialized model to LLaMA format with scaling and config adaptation.
"""

import os

# Convert a muP model to safetensors with muP scaling
os.system(
    "python -m model_converter_tool.cli convert path/to/mup_model --to safetensors --output path/to/output_model --mup2llama"
)

# You can also use other formats (e.g., gguf, hf, onnx) with --mup2llama
# os.system(
#     "python -m model_converter_tool.cli convert path/to/mup_model --to gguf --output path/to/output_model.gguf --mup2llama"
# )
