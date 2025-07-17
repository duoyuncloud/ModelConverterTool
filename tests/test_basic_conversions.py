#!/usr/bin/env python3
"""
Basic model format conversion tests
"""

import importlib.util
import os
import tempfile
from pathlib import Path
import sys
import platform
import requests
import numpy as np
import onnx
import onnxruntime
import pytest
from transformers import AutoModel, AutoTokenizer

from model_converter_tool.api import ModelConverterAPI

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

MODEL_NAME = "HuggingFaceM4/tiny-random-LlamaForCausalLM"

skip_mlx = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="MLX only supported on Apple Silicon macOS"
)

@pytest.fixture(scope="module")
def api():
    return ModelConverterAPI()

@pytest.fixture(scope="module")
def output_dir():
    d = Path("test_outputs/basic_conversions")
    d.mkdir(parents=True, exist_ok=True)
    return d

# Use dict to drive all README demos
DEMO_TASKS = [
    {"input_model": "gpt2", "output_format": "onnx", "output_file": "bert.onnx", "model_type": "feature-extraction"},
    # Use Qwen/Qwen2-0.5B for GGUF
    {"input_model": "Qwen/Qwen2-0.5B", "output_format": "gguf", "output_file": "qwen2-0.5b.gguf", "model_type": "text-generation"},
    {"input_model": "gpt2", "output_format": "mlx", "output_file": "gpt2.mlx", "model_type": "text-generation"},
    # fp16 is now tested as a safetensors variant
    {"input_model": "gpt2", "output_format": "safetensors", "output_file": "tiny_gpt2_fp16_safetensors", "model_type": "text-generation", "dtype": "fp16"},
    {"input_model": "bert-base-uncased", "output_format": "torchscript", "output_file": "bert.pt", "model_type": "feature-extraction"},
    {"input_model": "gpt2", "output_format": "safetensors", "output_file": "gpt2_safetensors", "model_type": "text-generation"},
    {"input_model": "gpt2", "output_format": "hf", "output_file": "gpt2_hf", "model_type": "text-generation"},
]

def is_hf_model_available(model_id):
    url = f"https://huggingface.co/{model_id}/resolve/main/config.json"
    try:
        r = requests.head(url, timeout=5, allow_redirects=True)
        return r.status_code == 200
    except Exception:
        return False

@pytest.mark.parametrize(
    "task",
    DEMO_TASKS,
    ids=[f"{t['input_model']}_to_{t['output_format']}" for t in DEMO_TASKS]
)
def test_basic_conversion(api, output_dir, task):
    # If it's an MLX task and not Apple Silicon, automatically skip
    if task["output_format"] == "mlx" and (platform.system() != "Darwin" or platform.machine() != "arm64"):
        pytest.skip("MLX only supported on Apple Silicon macOS")
    # GGUF conversion requires llama-cpp-python and a supported model family
    if task["output_format"] == "gguf":
        if not is_hf_model_available(task["input_model"]):
            pytest.skip(f"Model {task['input_model']} not available on HuggingFace")
    output_path = str(output_dir / task["output_file"])
    convert_kwargs = dict(
        model_name=task["input_model"],
        output_format=task["output_format"],
        output_path=output_path,
        model_type=task["model_type"],
        device="cpu",
    )
    if "dtype" in task:
        convert_kwargs["dtype"] = task["dtype"]
    result = api.convert_model(**convert_kwargs)
    assert result.success, f"{task['output_format']} conversion failed: {result.error}"
    assert os.path.exists(output_path)
