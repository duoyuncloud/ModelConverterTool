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

class TestQuantization:
    """Test quantization formats using opt-125m-local (local OPT-125M model)"""

    pytestmark = [pytest.mark.quantization, pytest.mark.slow]

    @pytest.fixture(autouse=True)
    def setup(self):
        self.converter = ModelConverter()
        self.test_model = "facebook/opt-125m"
        self.output_dir = Path("test_outputs/quantization")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def test_gptq_quantization(self):
        """Test GPTQ quantization"""
        result = self.converter.convert(
            input_source=self.test_model,
            output_format="gptq",
            output_path="outputs/opt_125m_gptq_quantized",
            model_type="text-generation",
            device="auto",
            validate=True,
            quantization_config={"damp_percent": 0.015},
        )
        assert result["success"], f"GPTQ quantization failed: {result.get('error')}"
        mv = result.get("model_validation", {})
        assert mv.get("success"), f"GPTQ model validation failed: {mv.get('error', 'No validation result')}"

    def test_awq_quantization(self):
        """Test AWQ quantization"""
        result = self.converter.convert(
            input_source=self.test_model,
            output_format="awq",
            output_path="outputs/opt_125m_awq_quantized",
            model_type="text-generation",
            device="auto",
            validate=True,
            quantization_config={"damp_percent": 0.015},
        )
        assert result["success"], f"AWQ quantization failed: {result.get('error')}"
        mv = result.get("model_validation", {})
        assert mv.get("success"), f"AWQ model validation failed: {mv.get('error', 'No validation result')}"

    def test_gguf_quantization(self):
        """Test GGUF quantization with different quantization levels"""
        quantization_levels = ["q4_k_m", "q8_0", "q5_k_m"]
        for quant_level in quantization_levels:
            output_path = str(self.output_dir / f"opt_125m_gguf_{quant_level}.gguf")
            result = self.converter.convert(
                input_source=self.test_model,
                output_format="gguf",
                output_path=output_path,
                model_type="text-generation",
                quantization=quant_level,
                device="auto",
                validate=True,
                quantization_config={"damp_percent": 0.015},
            )
            assert result["success"], f"GGUF quantization {quant_level} failed: {result.get('error', 'Unknown error')}"
            assert Path(output_path).exists(), f"GGUF output not found: {output_path}"

    @pytest.mark.parametrize("quant_type", ["gptq", "awq"])
    def test_cli_equivalent_quantization(self, quant_type):
        output_path = f"outputs/opt_125m_{quant_type}_cli"
        result = self.converter.convert(
            input_source=self.test_model,
            output_format=quant_type,
            output_path=output_path,
            model_type="text-generation",
            device="auto",
            validate=True,
            quantization_config={"damp_percent": 0.015},
        )
        assert result["success"], f"CLI equivalent quantization failed: {result.get('error')}"
        mv = result.get("model_validation", {})
        assert mv.get("success"), f"CLI equivalent quantization model validation failed: {mv.get('error', 'No validation result')}"
