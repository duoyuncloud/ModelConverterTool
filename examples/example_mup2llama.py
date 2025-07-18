"""
muP-to-LLaMA example: Convert a muP-initialized model to LLaMA format with scaling and config adaptation.
"""
import os

# Example command: convert a muP model to safetensors with muP scaling
# Replace 'path/to/mup_model' with your muP-initialized model directory
os.system(
    "python -m model_converter_tool.cli convert path/to/mup_model --to safetensors --output path/to/output_model --mup2llama"
)

# You can also use other formats (e.g., gguf, hf, onnx) with --mup2llama
# os.system(
#     "python -m model_converter_tool.cli convert path/to/mup_model --to gguf --output path/to/output_model.gguf --mup2llama"
# ) 