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

class TestQuantization:
    """Test quantization formats using tiny-gpt2"""

    pytestmark = [pytest.mark.quantization, pytest.mark.slow]

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        self.converter = ModelConverter()
        self.test_model = "opt-125m-local"  # 改为本地 OPT-125M 模型
        self.output_dir = Path("test_outputs/quantization")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def test_gptq_quantization(self):
        """Test GPTQ quantization"""
        result = self.converter.convert(
            input_source="opt-125m-local",  # 本地 OPT-125M 模型
            output_format="gptq",
            output_path="outputs/opt_125m_gptq_quantized",
            model_type="text-generation",
            device="auto",  # 自动选择GPU或CPU
            validate=True,
        )
        assert result["success"], f"GPTQ quantization failed: {result.get('error')}"
        mv = result.get("model_validation", {})
        assert mv.get("success"), f"GPTQ model validation failed: {mv.get('error', 'No validation result')}"

    def test_awq_quantization(self):
        """Test AWQ quantization"""
        result = self.converter.convert(
            input_source="opt-125m-local",  # 本地 OPT-125M 模型
            output_format="awq",
            output_path="outputs/opt_125m_awq_quantized",
            model_type="text-generation",
            device="auto",  # 自动选择GPU或CPU
            validate=True,
        )
        assert result["success"], f"AWQ quantization failed: {result.get('error')}"
        mv = result.get("model_validation", {})
        assert mv.get("success"), f"AWQ model validation failed: {mv.get('error', 'No validation result')}"

    def test_gguf_quantization(self):
        """Test GGUF quantization with different quantization levels"""
        quantization_levels = ["q4_k_m", "q8_0", "q5_k_m"]

        for quant_level in quantization_levels:
            output_path = str(self.output_dir / f"tiny_gpt2_gguf_{quant_level}.gguf")
            result = self.converter.convert(
                input_source=self.test_model,
                output_format="gguf",
                output_path=output_path,
                model_type="text-generation",
                quantization=quant_level,
                device="auto",
                validate=True,
            )
            assert result[
                "success"
            ], f"GGUF quantization {quant_level} failed: {result.get('error', 'Unknown error')}"
            assert os.path.exists(output_path), f"GGUF output not found: {output_path}"

    @pytest.mark.parametrize("quant_type", ["gptq", "awq"])
    def test_cli_equivalent_quantization(self, quant_type):
        output_path = f"outputs/opt_125m_{quant_type}_cli"
        result = self.converter.convert(
            input_source="opt-125m-local",  # 本地 OPT-125M 模型
            output_format=quant_type,
            output_path=output_path,
            model_type="text-generation",
            device="auto",  # 自动选择GPU或CPU
            validate=True,
        )
        assert result["success"], f"CLI equivalent quantization failed: {result.get('error')}"
        mv = result.get("model_validation", {})
        assert mv.get("success"), f"CLI equivalent quantization model validation failed: {mv.get('error', 'No validation result')}"
