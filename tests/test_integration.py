import subprocess
import sys
import os
import pytest
import yaml
from pathlib import Path
from model_converter_tool.api import ModelConverterAPI

CLI_CMD = [sys.executable, "-m", "model_converter_tool.cli"]

@pytest.fixture(scope="module")
def tmp_output(tmp_path_factory):
    d = tmp_path_factory.mktemp("integration_outputs")
    return d

# --- API and CLI single conversion integration ---
def test_api_and_cli_conversion(tmp_output):
    """
    Test both API and CLI conversion for a representative model/format.
    """
    api = ModelConverterAPI()
    output_path = tmp_output / "gpt2.onnx"
    # API conversion
    result = api.convert_model(
        model_path="gpt2",
        output_format="onnx",
        output_path=str(output_path),
        model_type="text-generation",
        device="cpu"
    )
    assert result.success, f"API conversion failed: {result.error}"
    assert output_path.exists()
    # CLI conversion
    cli_output = tmp_output / "cli_gpt2.onnx"
    cli_result = subprocess.run(
        CLI_CMD + ["convert", "gpt2", "onnx", "-o", str(cli_output)],
        capture_output=True, text=True
    )
    assert cli_result.returncode == 0
    assert "Conversion succeeded" in cli_result.stdout
    assert cli_output.exists()

# --- Batch conversion integration ---
def test_batch_conversion(tmp_output):
    """
    Test batch conversion via CLI with a temporary YAML config file.
    """
    batch_config = {
        "tasks": [
            {
                "model_path": "gpt2",
                "output_format": "onnx",
                "output_path": str(tmp_output / "batch_gpt2.onnx"),
                "model_type": "text-generation"
            },
            {
                "model_path": "facebook/opt-125m",
                "output_format": "gptq",
                "output_path": str(tmp_output / "batch_opt125m_gptq"),
                "model_type": "text-generation",
                "quantization": "4bit"
            }
        ]
    }
    config_path = tmp_output / "batch_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(batch_config, f)
    result = subprocess.run(
        CLI_CMD + ["batch", str(config_path)],
        capture_output=True, text=True
    )
    assert result.returncode in (0, 2)  # 0: all success, 2: some failed
    assert "Batch conversion completed" in result.stdout
    assert (tmp_output / "batch_gpt2.onnx").exists()

# --- Config, history, cache integration ---
def test_config_show_and_set():
    """
    Test config show and set via CLI.
    """
    result = subprocess.run(CLI_CMD + ["config", "show"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "cache_dir" in result.stdout or "config" in result.stdout.lower()
    # Set a config value
    result2 = subprocess.run(CLI_CMD + ["config", "set", "test_key", "test_value"], capture_output=True, text=True)
    assert result2.returncode == 0
    assert "test_key" in result2.stdout

def test_history():
    """
    Test history commands via CLI.
    """
    result = subprocess.run(CLI_CMD + ["history"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Completed tasks" in result.stdout

# --- Error and edge case integration ---
def test_invalid_format_error(tmp_output):
    """
    Test CLI convert with unsupported format (should fail gracefully).
    """
    output_path = tmp_output / "invalid_format.out"
    result = subprocess.run(
        CLI_CMD + ["convert", "gpt2", "invalidformat", "-o", str(output_path)],
        capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "unsupported output format" in result.stdout.lower() or "validation failed" in result.stdout.lower() or "failed" in result.stdout.lower()

# --- Help/version integration ---
def test_cli_main_help_and_version():
    """
    Test CLI main --help and --version output.
    """
    help_result = subprocess.run(CLI_CMD + ["--help"], capture_output=True, text=True)
    assert help_result.returncode == 0
    assert "usage" in help_result.stdout.lower() or "help" in help_result.stdout.lower()
    version_result = subprocess.run(CLI_CMD + ["--version"], capture_output=True, text=True)
    assert version_result.returncode == 0
    assert "version" in version_result.stdout.lower() 