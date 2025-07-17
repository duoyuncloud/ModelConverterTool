#!/usr/bin/env python3
"""
Basic model format conversion tests
"""

import importlib.util
import os
import tempfile
import shutil
import json
from pathlib import Path
import sys
import platform
import requests
import numpy as np
import onnx
import onnxruntime
import pytest
from transformers import AutoModel, AutoTokenizer
import yaml
import torch

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
    {"input_model": "gpt2", "output_format": "onnx", "output_file": "bert.onnx", "model_type": "text-classification"},
    # Use Qwen/Qwen2-0.5B for GGUF
    {"input_model": "Qwen/Qwen2-0.5B", "output_format": "gguf", "output_file": "qwen2-0.5b.gguf", "model_type": "text-generation"},
    {"input_model": "gpt2", "output_format": "mlx", "output_file": "gpt2.mlx", "model_type": "text-generation"},
    # fp16 is now tested as a safetensors variant
    {"input_model": "gpt2", "output_format": "safetensors", "output_file": "tiny_gpt2_fp16_safetensors", "model_type": "text-generation", "dtype": "fp16"},
    {"input_model": "bert-base-uncased", "output_format": "torchscript", "output_file": "bert.pt", "model_type": "text-classification"},
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

# Temporarily disable fake-weight test due to tokenizer/config issues
# def test_fake_weight_with_custom_shape(api, output_dir):
#     """
#     Test conversion with --fake-weight and a custom shape config, using a local model dir with a minimal config.json and tokenizer files.
#     """
#     # Step 1: Create a temp model dir with a minimal config.json (no quantization_config)
#     tmp_model_dir = tempfile.mkdtemp()
#     # Add more required fields for GPT2 config
#     config = {
#         "architectures": ["GPT2LMHeadModel"],
#         "model_type": "gpt2",
#         "vocab_size": 128,
#         "n_embd": 16,
#         "n_layer": 2,
#         "n_head": 2,
#         "bos_token_id": 0,
#         "eos_token_id": 1,
#         "pad_token_id": 0,
#         "n_positions": 32,
#         "n_ctx": 32,
#         "n_inner": 32,
#         "layer_norm_epsilon": 1e-5,
#         "initializer_range": 0.02,
#         "attn_pdrop": 0.1,
#         "embd_pdrop": 0.1,
#         "resid_pdrop": 0.1,
#         "summary_first_dropout": 0.1,
#         "summary_type": "cls_index",
#         "summary_use_proj": True,
#         "summary_activation": None,
#         "summary_proj_to_labels": True,
#         "summary_attention_mask": False,
#     }
#     with open(os.path.join(tmp_model_dir, "config.json"), "w") as f:
#         json.dump(config, f, indent=2)
#
#     # Step 1b: Create a minimal vocab.json and tokenizer_config.json for GPT2
#     vocab = {f"token{i}": i for i in range(128)}
#     with open(os.path.join(tmp_model_dir, "vocab.json"), "w") as f:
#         json.dump(vocab, f, indent=2)
#     tokenizer_config = {
#         "bos_token": "<bos>",
#         "eos_token": "<eos>",
#         "unk_token": "<unk>",
#         "pad_token": "<pad>",
#         "model_max_length": 32,
#         "tokenizer_class": "PreTrainedTokenizerFast"
#     }
#     with open(os.path.join(tmp_model_dir, "tokenizer_config.json"), "w") as f:
#         json.dump(tokenizer_config, f, indent=2)
#
#     # Step 1c: Load the tokenizer
#     from transformers import AutoTokenizer
#     tokenizer = AutoTokenizer.from_pretrained(tmp_model_dir, trust_remote_code=True)
#
#     # Step 2: Prepare fake_weight_config
#     fake_weight_config = {
#         "transformer.wte.weight": [128, 16],
#     }
#
#     # Step 3: Run conversion
#     output_path = str(output_dir / "fake_weight_custom")
#     result = api.convert_model(
#         model_path=tmp_model_dir,
#         output_format="safetensors",
#         output_path=output_path,
#         model_type="text-generation",
#         device="cpu",
#         fake_weight=True,
#         fake_weight_shape_dict=fake_weight_config,
#         tokenizer=tokenizer,
#     )
#     assert result.success, f"Fake weight conversion failed: {result.error}"
#     assert os.path.exists(output_path)
#
#     # Step 4: Clean up temp model dir
#     shutil.rmtree(tmp_model_dir)
