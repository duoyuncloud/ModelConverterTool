#!/usr/bin/env python3
"""
Quantization format tests
All tests use sshleifer/tiny-gpt2 for faster testing
"""

import os
import sys
from pathlib import Path

import pytest
import torch

from model_converter_tool.converter import ModelConverter


class TestQuantization:
    """Test quantization formats using tiny-gpt2"""

    pytestmark = [pytest.mark.quantization, pytest.mark.slow]

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        self.converter = ModelConverter()
        self.test_model = "sshleifer/tiny-gpt2"  # Use tiny model for faster testing
        self.output_dir = Path("test_outputs/quantization")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def test_gptq_quantization(self):
        """Test GPTQ quantization"""
        model_name = "sshleifer/tiny-gpt2"
        output_path = "./outputs/tiny_gpt2_gptq"
        result = self.converter.convert(
            input_source=model_name,
            output_format="gptq",
            output_path=output_path,
            model_type="text-generation",
            device="cpu",
            validate=True
        )
        print("GPTQ model_validation:", result.get("model_validation"))
        assert result["success"], f"GPTQ quantization failed: {result.get('error', 'Unknown error')}"
        mv = result.get("model_validation", {})
        assert mv.get("success") or (
            "not available" in str(mv.get("error", "")).lower()
            or "unsupported" in str(mv.get("error", "")).lower()
            or "基础验证" in str(mv.get("error", ""))
        ), f"GPTQ model validation failed: {mv.get('error', 'No validation result')}"

    def test_awq_quantization(self):
        """Test AWQ quantization"""
        model_name = "sshleifer/tiny-gpt2"
        output_path = "./outputs/tiny_gpt2_awq"
        result = self.converter.convert(
            input_source=model_name,
            output_format="awq",
            output_path=output_path,
            model_type="text-generation",
            device="cpu",
            validate=True
        )
        print("AWQ model_validation:", result.get("model_validation"))
        assert result["success"], f"AWQ quantization failed: {result.get('error', 'Unknown error')}"
        mv = result.get("model_validation", {})
        assert mv.get("success") or (
            "not available" in str(mv.get("error", "")).lower()
            or "unsupported" in str(mv.get("error", "")).lower()
            or "基础验证" in str(mv.get("error", ""))
        ), f"AWQ model validation failed: {mv.get('error', 'No validation result')}"

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
                device="cpu",
                validate=True,
            )
            assert result[
                "success"
            ], f"GGUF quantization {quant_level} failed: {result.get('error', 'Unknown error')}"
            assert os.path.exists(output_path), f"GGUF output not found: {output_path}"

    @pytest.mark.parametrize("output_format", ["gptq", "awq"])
    def test_cli_equivalent_quantization(self, output_format):
        model_name = self.test_model
        output_dir = self.output_dir
        output_path = output_dir / f"{model_name.replace('/', '_')}.{output_format}"
        result = self.converter.convert(
            input_source=model_name,
            output_format=output_format,
            output_path=str(output_path),
            model_type="text-generation",
            device="cpu",
            validate=True,
        )
        assert result["success"], f"{output_format} quantization failed: {result.get('error')}"
        assert os.path.exists(output_path), f"{output_format} output file not found: {output_path}"
