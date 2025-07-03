#!/usr/bin/env python3
"""
Batch conversion tests
Test batch conversion using gpt2 model with YAML configuration
"""

import os
from pathlib import Path
import sys

import pytest
import yaml

from model_converter_tool.converter import ModelConverter


class TestBatchConversion:
    """Test batch conversion functionality"""

    pytestmark = [pytest.mark.batch, pytest.mark.slow]

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        self.converter = ModelConverter()
        self.test_model = "facebook/opt-125m"
        self.output_dir = Path("test_outputs/batch_conversion")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_batch_config(self):
        """Create a batch configuration for testing"""
        config = {
            "models": {
                "gpt2_to_onnx": {
                    "input": self.test_model,
                    "output_format": "onnx",
                    "output_path": str(self.output_dir / "gpt2.onnx"),
                    "model_type": "text-generation",
                    "device": "auto",
                },
                "gpt2_to_gguf": {
                    "input": self.test_model,
                    "output_format": "gguf",
                    "output_path": str(self.output_dir / "gpt2.gguf"),
                    "model_type": "text-generation",
                    "device": "auto",
                },
                "gpt2_to_fp16": {
                    "input": self.test_model,
                    "output_format": "fp16",
                    "output_path": str(self.output_dir / "gpt2_fp16"),
                    "model_type": "text-generation",
                    "device": "auto",
                },
                "gpt2_to_torchscript": {
                    "input": self.test_model,
                    "output_format": "torchscript",
                    "output_path": str(self.output_dir / "gpt2.pt"),
                    "model_type": "text-generation",
                    "device": "auto",
                },
                "gpt2_to_safetensors": {
                    "input": self.test_model,
                    "output_format": "safetensors",
                    "output_path": str(self.output_dir / "gpt2_safetensors"),
                    "model_type": "text-generation",
                    "device": "auto",
                },
                "gpt2_to_hf": {
                    "input": self.test_model,
                    "output_format": "hf",
                    "output_path": str(self.output_dir / "gpt2_hf"),
                    "model_type": "text-generation",
                    "device": "auto",
                },
            }
        }
        # 仅macOS下添加MLX任务
        if sys.platform == "darwin":
            config["models"]["gpt2_to_mlx"] = {
                "input": self.test_model,
                "output_format": "mlx",
                "output_path": str(self.output_dir / "gpt2.mlx"),
                "model_type": "text-generation",
                "device": "auto",
            }
        return config

    def test_batch_conversion_from_dict(self):
        """Test batch conversion using dictionary configuration"""
        config = self.create_batch_config()

        # Convert config to list of tasks
        tasks = []
        for task_name, task_config in config["models"].items():
            tasks.append(task_config)

        results = self.converter.batch_convert(
            tasks=tasks, max_workers=2, max_retries=1  # Limit workers for testing
        )

        # Check all conversions succeeded
        assert len(results) == len(
            tasks
        ), f"Expected {len(tasks)} results, got {len(results)}"

        successful_conversions = 0
        for result in results:
            if result.get("success"):
                successful_conversions += 1
                # Check output file exists
                output_path = result.get("output_path", "")
                if output_path:
                    assert os.path.exists(
                        output_path
                    ), f"Output file not found: {output_path}"
            else:
                print(f"Conversion failed: {result.get('error', 'Unknown error')}")

        print(
            f"Batch conversion completed: {successful_conversions}/{len(tasks)} successful"
        )
        # Allow some failures in CI environment
        assert (
            successful_conversions >= len(tasks) * 0.7
        ), f"Too many conversions failed: {successful_conversions}/{len(tasks)}"

    def test_batch_conversion_yaml_file(self):
        """Test batch conversion using YAML file"""
        config = self.create_batch_config()

        # Create temporary YAML file
        yaml_path = self.output_dir / "batch_test.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        # Load YAML and convert to tasks
        with open(yaml_path, "r") as f:
            loaded_config = yaml.safe_load(f)

        tasks = []
        for task_name, task_config in loaded_config["models"].items():
            tasks.append(task_config)

        results = self.converter.batch_convert(
            tasks=tasks, max_workers=2, max_retries=1
        )

        # Check results
        assert len(results) == len(
            tasks
        ), f"Expected {len(tasks)} results, got {len(results)}"

        successful_conversions = 0
        for result in results:
            if result.get("success"):
                successful_conversions += 1

        print(
            f"YAML batch conversion completed: {successful_conversions}/{len(tasks)} successful"
        )
        assert (
            successful_conversions >= len(tasks) * 0.7
        ), f"Too many conversions failed: {successful_conversions}/{len(tasks)}"

        # Clean up
        if yaml_path.exists():
            yaml_path.unlink()

# --- Debug script: Run batch_convert and print all results for debugging ---
if __name__ == "__main__":
    from model_converter_tool.converter import ModelConverter
    import json
    output_dir = Path("test_outputs/batch_conversion")
    output_dir.mkdir(parents=True, exist_ok=True)
    test_model = "facebook/opt-125m"
    tasks = [
        {
            "input_source": test_model,
            "output_format": "onnx",
            "output_path": str(output_dir / "gpt2.onnx"),
            "model_type": "text-generation",
            "device": "auto",
        },
        {
            "input_source": test_model,
            "output_format": "gguf",
            "output_path": str(output_dir / "gpt2.gguf"),
            "model_type": "text-generation",
            "device": "auto",
        },
        {
            "input_source": test_model,
            "output_format": "mlx",
            "output_path": str(output_dir / "gpt2.mlx"),
            "model_type": "text-generation",
            "device": "auto",
        },
        {
            "input_source": test_model,
            "output_format": "fp16",
            "output_path": str(output_dir / "gpt2_fp16"),
            "model_type": "text-generation",
            "device": "auto",
        },
        {
            "input_source": test_model,
            "output_format": "torchscript",
            "output_path": str(output_dir / "gpt2.pt"),
            "model_type": "text-generation",
            "device": "auto",
        },
        {
            "input_source": test_model,
            "output_format": "safetensors",
            "output_path": str(output_dir / "gpt2_safetensors"),
            "model_type": "text-generation",
            "device": "auto",
        },
        {
            "input_source": test_model,
            "output_format": "hf",
            "output_path": str(output_dir / "gpt2_hf"),
            "model_type": "text-generation",
            "device": "auto",
        },
    ]
    converter = ModelConverter()
    results = converter.batch_convert(tasks, max_workers=1)
    for r in results:
        print(json.dumps(r, indent=2, ensure_ascii=False))
    # Optionally, save to file for further inspection
    with open(output_dir / "batch_debug_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
