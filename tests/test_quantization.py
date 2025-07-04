import multiprocessing as mp
mp.set_start_method("spawn", force=True)

print("=== DEBUG: test_quantization.py started ===")
import importlib
print("find_spec('gptqmodel'):", importlib.util.find_spec("gptqmodel"))
try:
    import gptqmodel
    print("gptqmodel version:", gptqmodel.__version__)
except Exception as e:
    print("gptqmodel import error:", e)

#!/usr/bin/env python3
"""
Quantization format tests
All tests use sshleifer/tiny-gpt2 for faster testing
"""

import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# os.environ["MPS_VISIBLE_DEVICES"] = ""
# os.environ["USE_CPU_ONLY"] = "1"

import sys
from pathlib import Path

import pytest
import torch
import importlib

from model_converter_tool.converter import ModelConverter

# Skip quantization tests if gptqmodel is missing
# skip_quant = importlib.util.find_spec("gptqmodel") is None
# pytestmark = pytest.mark.skipif(
#     skip_quant,
#     reason="gptqmodel not available"
# )

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

@pytest.mark.parametrize("input_model,output_format,output_file,quantization", [
    ("facebook/opt-125m", "gptq", "opt_125m_gptq", "q4_k_m"),
    ("facebook/opt-125m", "awq", "opt_125m_awq", "q4_k_m"),
])
def test_quantization(converter, output_dir, input_model, output_format, output_file, quantization):
    output_path = str(output_dir / output_file)
    result = converter.convert(
        input_source=input_model,
        output_format=output_format,
        output_path=output_path,
        model_type="text-generation",
        device="cpu",
        quantization=quantization,
        validate=True,
    )
    assert result["success"], f"{output_format} quantization failed: {result.get('error')}"
    assert os.path.exists(output_path)
