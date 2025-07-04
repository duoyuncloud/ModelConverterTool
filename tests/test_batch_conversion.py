import os
import pytest
import yaml
from model_converter_tool.converter import ModelConverter
from pathlib import Path

@pytest.fixture(scope="module")
def converter():
    return ModelConverter()

@pytest.fixture(scope="module")
def output_dir():
    d = Path("test_outputs/batch_conversion")
    d.mkdir(parents=True, exist_ok=True)
    return d

def test_batch_conversion_from_yaml(converter, output_dir):
    batch_config = {
        "models": [
            {
                "model_name": "bert-base-uncased",
                "output_format": "onnx",
                "output_path": str(output_dir / "bert.onnx"),
                "model_type": "feature-extraction",
                "device": "cpu",
            },
            {
                "model_name": "sshleifer/tiny-gpt2",
                "output_format": "fp16",
                "output_path": str(output_dir / "tiny_gpt2_fp16"),
                "model_type": "text-generation",
                "device": "cpu",
            },
            {
                "model_name": "bert-base-uncased",
                "output_format": "torchscript",
                "output_path": str(output_dir / "bert.pt"),
                "model_type": "feature-extraction",
                "device": "cpu",
            },
        ]
    }
    yaml_path = output_dir / "batch_test.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(batch_config, f, default_flow_style=False)
    with open(yaml_path, "r") as f:
        tasks = yaml.safe_load(f)["models"]
    results = converter.batch_convert(tasks)
    for i, task in enumerate(batch_config["models"]):
        if not os.path.exists(task["output_path"]):
            print(f"[DEBUG] Task {i} failed: {results[i].error}")
        assert os.path.exists(task["output_path"]), f"Output not found: {task['output_path']}"