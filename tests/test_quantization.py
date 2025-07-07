#!/usr/bin/env python3
"""
Quantization format tests
All tests use facebook/opt-125m for faster testing
"""

import os
import sys
from pathlib import Path

import pytest
import torch

from model_converter_tool.converter import ModelConverter

torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built = lambda: False

# 在CI环境下跳过macOS量化测试
is_ci = os.environ.get('CI') == 'true' or os.environ.get('GITHUB_ACTIONS') == 'true'
if sys.platform == 'darwin' and is_ci:
    pytest.skip('Skip quantization tests on macOS CI due to known ABI/runner issues', allow_module_level=True)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture(scope="module")
def converter():
    return ModelConverter()

@pytest.fixture(scope="module")
def output_dir():
    d = Path("test_outputs/quantization")
    d.mkdir(parents=True, exist_ok=True)
    return d

@pytest.mark.parametrize("input_model,output_format,output_file,quantization,model_type", [
    ("facebook/opt-125m", "gptq", "opt_125m_gptq", "4bit", "text-generation"),
    ("facebook/opt-125m", "awq", "opt_125m_awq", "4bit", "text-generation"),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "gguf", "tinyllama-1.1b-chat-v1.0.gguf", "q4_k_m", "text-generation"),
    ("gpt2", "mlx", "gpt2.mlx", "q4_k_m", "text-generation"),
])
def test_quantization(converter, output_dir, input_model, output_format, output_file, quantization, model_type):
    output_path = str(output_dir / output_file)
    result = converter.convert(
        model_name=input_model,
        output_format=output_format,
        output_path=output_path,
        model_type=model_type,
        device="cpu",
        quantization=quantization,
    )
    print(f"DEBUG: result.success = {result.success}")
    print(f"DEBUG: result.error = {result.error}")
    print(f"DEBUG: result.validation = {result.validation}")
    print(f"DEBUG: result.output_path = {result.output_path}")
    assert result.success, f"{output_format} quantization failed: {result.error}"
    assert os.path.exists(output_path) 