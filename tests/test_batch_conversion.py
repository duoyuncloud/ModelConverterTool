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
        "conversions": [
            {
                "input": "bert-base-uncased",
                "output": str(output_dir / "bert.onnx"),
                "to": "onnx",
                "model_type": "feature-extraction",
                "device": "cpu",
            },
            {
                "input": "sshleifer/tiny-gpt2",
                "output": str(output_dir / "tiny_gpt2_fp16"),
                "to": "fp16",
                "model_type": "text-generation",
                "device": "cpu",
            },
            {
                "input": "bert-base-uncased",
                "output": str(output_dir / "bert.pt"),
                "to": "torchscript",
                "model_type": "feature-extraction",
                "device": "cpu",
            },
        ]
    }
    yaml_path = output_dir / "batch_test.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(batch_config, f, default_flow_style=False)
    with open(yaml_path, "r") as f:
        tasks = yaml.safe_load(f)["conversions"]
    results = converter.batch_convert(tasks)
    for i, task in enumerate(batch_config["conversions"]):
        if not os.path.exists(task["output"]):
            print(f"[DEBUG] Task {i} failed: {results[i].error}")
        assert os.path.exists(task["output"]), f"Output not found: {task['output']}"