#!/usr/bin/env python3
"""
API usage tests
Test ModelConverter().convert and ModelConverter().batch_convert methods
"""

import os
from pathlib import Path
import sys
import torch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model_converter_tool.converter import ModelConverter


class TestAPIUsage:
    """Test API usage examples"""

    pytestmark = [pytest.mark.api, pytest.mark.fast]

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        self.converter = ModelConverter()
        self.test_model = "gpt2"
        self.output_dir = Path("test_outputs/api_usage")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def test_convert_method_basic(self):
        """Test ModelConverter().convert basic usage"""
        output_path = str(self.output_dir / "gpt2_onnx.onnx")

        result = self.converter.convert(
            input_source=self.test_model,
            output_format="onnx",
            output_path=output_path,
            model_type="text-generation",
            device="auto",
            validate=True,
        )

        assert result[
            "success"
        ], f"Convert method failed: {result.get('error', 'Unknown error')}"
        assert os.path.exists(output_path), f"Output file not found: {output_path}"

    def test_convert_method_all_formats(self):
        """Test ModelConverter().convert with all supported formats"""
        formats = ["onnx", "gguf", "mlx", "fp16", "torchscript", "safetensors", "hf"]

        for format_name in formats:
            if format_name == "hf":
                output_path = str(self.output_dir / f"gpt2_{format_name}")
            elif format_name in ["fp16", "safetensors"]:
                output_path = str(self.output_dir / f"gpt2_{format_name}")
            else:
                output_path = str(self.output_dir / f"gpt2.{format_name}")

            result = self.converter.convert(
                input_source=self.test_model,
                output_format=format_name,
                output_path=output_path,
                model_type="text-generation",
                device="auto",
                validate=True,
            )

            if result["success"]:
                assert os.path.exists(
                    output_path
                ), f"Output file not found: {output_path}"
            else:
                print(
                    f"Format {format_name} failed: {result.get('error', 'Unknown error')}"
                )

    def test_batch_convert_method(self):
        formats = ["onnx", "gguf", "fp16", "torchscript", "hf"]
        tasks = []
        for fmt in formats:
            output_path = str(self.output_dir / f"batch_gpt2_{fmt}")
            tasks.append({
                "input_source": self.test_model,
                "output_format": fmt,
                "output_path": output_path,
                "model_type": "text-generation",
                "device": "auto",
                "validate": True,
            })
        results = self.converter.batch_convert(tasks=tasks, max_workers=1, max_retries=1)
        for result in results:
            assert result.get("success"), f"Batch {result.get('output_format')} failed: {result.get('error')}"
            assert os.path.exists(result.get("output_path", "")), f"Batch {result.get('output_format')} output file not found: {result.get('output_path')}"

    def test_convert_with_quantization(self):
        for fmt in ["gptq", "awq"]:
            output_path = str(self.output_dir / f"opt_125m_{fmt}")
            result = self.converter.convert(
                input_source="facebook/opt-125m",
                output_format=fmt,
                output_path=output_path,
                model_type="text-generation",
                device="auto",
                validate=True,
            )
            assert result["success"], f"{fmt} quantization failed: {result.get('error')}"
            assert os.path.exists(output_path), f"{fmt} output file not found: {output_path}"

    def test_convert_with_validation(self):
        """Test ModelConverter().convert with validation enabled"""
        output_path = str(self.output_dir / "gpt2_validated.onnx")

        result = self.converter.convert(
            input_source=self.test_model,
            output_format="onnx",
            output_path=output_path,
            model_type="text-generation",
            device="auto",
            validate=True,  # Enable validation
        )

        assert result[
            "success"
        ], f"Conversion with validation failed: {result.get('error', 'Unknown error')}"
        assert result.get("validation", False), "Validation should be enabled"
        assert os.path.exists(output_path), f"Output file not found: {output_path}"

    def test_convert_without_validation(self):
        """Test ModelConverter().convert without validation"""
        output_path = str(self.output_dir / "gpt2_no_validation.onnx")

        result = self.converter.convert(
            input_source=self.test_model,
            output_format="onnx",
            output_path=output_path,
            model_type="text-generation",
            device="auto",
            validate=False,  # Disable validation
        )

        assert result[
            "success"
        ], f"Conversion without validation failed: {result.get('error', 'Unknown error')}"
        assert os.path.exists(output_path), f"Output file not found: {output_path}"
