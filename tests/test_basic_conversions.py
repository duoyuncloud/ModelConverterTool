#!/usr/bin/env python3
"""
Basic model format conversion tests
"""

import os
import platform
import pytest
from conftest import is_hf_model_available

MODEL_NAME = "HuggingFaceM4/tiny-random-LlamaForCausalLM"

skip_mlx = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64", reason="MLX only supported on Apple Silicon macOS"
)

# Use dict to drive all README demos
DEMO_TASKS = [
    {"input_model": "gpt2", "output_format": "onnx", "output_file": "bert.onnx", "model_type": "text-classification"},
    # Use Qwen/Qwen2-0.5B for GGUF
    {
        "input_model": "Qwen/Qwen2-0.5B",
        "output_format": "gguf",
        "output_file": "qwen2-0.5b.gguf",
        "model_type": "text-generation",
    },
    {"input_model": "gpt2", "output_format": "mlx", "output_file": "gpt2.mlx", "model_type": "text-generation"},
    # fp16 is now tested as a safetensors variant
    {
        "input_model": "gpt2",
        "output_format": "safetensors",
        "output_file": "tiny_gpt2_fp16_safetensors",
        "model_type": "text-generation",
        "dtype": "fp16",
    },
    {
        "input_model": "bert-base-uncased",
        "output_format": "torchscript",
        "output_file": "bert.pt",
        "model_type": "text-classification",
    },
    {
        "input_model": "gpt2",
        "output_format": "safetensors",
        "output_file": "gpt2_safetensors",
        "model_type": "text-generation",
    },
    {"input_model": "gpt2", "output_format": "hf", "output_file": "gpt2_hf", "model_type": "text-generation"},
]


@pytest.mark.parametrize(
    "task",
    DEMO_TASKS,
    ids=[
        f"{t['input_model']}_to_{t['output_format']}{'_fp16' if t.get('dtype') == 'fp16' else ''}" for t in DEMO_TASKS
    ],
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
        model_path=task["input_model"],  # Use model_path as required by API
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
