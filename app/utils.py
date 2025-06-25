"""
utils.py
Shared helper functions and constants for the backend.
"""

import os
import json
from fastapi import HTTPException

# Format extension mapping (only supported formats)
FORMAT_EXTENSIONS = {
    "onnx": ".onnx",
    "torchscript": ".pt",
    "fp16": "_fp16",
    "hf": "_hf",
    "gptq": "_gptq",
    "awq": "_awq",
    "gguf": ".gguf",
    "mlx": ".npz",
    "test": ".json"
}

def parse_json_params(params: str, error_msg: str = "Invalid parameter format") -> dict:
    """Parse and validate parameters from a JSON string."""
    try:
        return json.loads(params)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail=error_msg)

def generate_output_path(file_id: str, target_format: str, output_dir: str) -> str:
    """Generate output file path based on format and file ID."""
    if target_format.lower() == "onnx":
        # For ONNX, use a directory, not a file with .onnx extension
        return os.path.join(output_dir, file_id)
    output_ext = FORMAT_EXTENSIONS.get(target_format.lower(), "")
    return os.path.join(output_dir, f"{file_id}{output_ext}") 