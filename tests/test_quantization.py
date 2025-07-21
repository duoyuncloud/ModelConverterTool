#!/usr/bin/env python3
"""
Quantization format tests
All tests use facebook/opt-125m for faster testing
"""

import os
import sys
from pathlib import Path
import platform

import pytest

from model_converter_tool.api import ModelConverterAPI

# Skip macOS quantization tests in CI environment
is_ci = os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true"
if sys.platform == "darwin" and is_ci:
    pytest.skip("Skip quantization tests on macOS CI due to known ABI/runner issues", allow_module_level=True)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture(scope="module")
def api():
    return ModelConverterAPI()


@pytest.fixture(scope="module")
def output_dir():
    d = Path("test_outputs/quantization")
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.mark.parametrize(
    "input_model,output_format,output_file,quantization,model_type",
    [
        ("facebook/opt-125m", "gptq", "opt_125m_gptq", "4bit", "text-generation"),
        ("facebook/opt-125m", "awq", "opt_125m_awq", "4bit", "text-generation"),
        ("Qwen/Qwen2-0.5B", "gguf", "qwen2-0.5b.gguf", "q4_k_m", "text-generation"),
        ("gpt2", "mlx", "gpt2.mlx", "q4_k_m", "text-generation"),
    ],
    ids=[
        f"{m}_to_{f}_{q}"
        for m, f, _, q, _ in [
            ("facebook/opt-125m", "gptq", "opt_125m_gptq", "4bit", "text-generation"),
            ("facebook/opt-125m", "awq", "opt_125m_awq", "4bit", "text-generation"),
            ("Qwen/Qwen2-0.5B", "gguf", "qwen2-0.5b.gguf", "q4_k_m", "text-generation"),
            ("gpt2", "mlx", "gpt2.mlx", "q4_k_m", "text-generation"),
        ]
    ],
)
def test_quantization(api, output_dir, input_model, output_format, output_file, quantization, model_type):
    # Automatically skip MLX tests (non-Apple Silicon)
    if output_format == "mlx" and (platform.system() != "Darwin" or platform.machine() != "arm64"):
        pytest.skip("MLX only supported on Apple Silicon macOS")

    output_path = str(output_dir / output_file)
    result = api.convert_model(
        model_path=input_model,
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


def test_quantization_config_applied(api, output_dir):
    import json
    import os

    quant_config = {"bits": 3, "group_size": 64, "sym": True, "desc": "test-desc"}
    output_path = str(output_dir / "opt_125m_gptq_custom")
    result = api.convert_model(
        model_path="facebook/opt-125m",
        output_format="gptq",
        output_path=output_path,
        model_type="text-generation",
        device="cpu",
        quantization_config=quant_config,
    )
    assert result.success, f"Quantization with config failed: {result.error}"
    config_path = os.path.join(output_path, "config.json")
    assert os.path.exists(config_path), "config.json not found in output"
    with open(config_path, "r") as f:
        config = json.load(f)
    qcfg = config.get("quantization_config", {})
    for k, v in quant_config.items():
        assert qcfg.get(k) == v, f"Quantization param {k} not applied: expected {v}, got {qcfg.get(k)}"
