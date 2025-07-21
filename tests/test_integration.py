import subprocess
import sys
import os
import pytest
import yaml
from pathlib import Path
from model_converter_tool.api import ModelConverterAPI
import tempfile
import shutil

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
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        api_output = tmpdir / "api_gpt2_onnx"
        cli_output = tmpdir / "cli_gpt2_onnx"
        # API conversion
        api = ModelConverterAPI()
        result = api.convert_model(
            model_path="sshleifer/tiny-gpt2",
            output_format="onnx",
            output_path=str(api_output),
        )
        assert result.success
        # Check for model.onnx in the output directory
        assert (api_output / "model.onnx").exists()
        # CLI conversion
        subprocess.check_call([
            sys.executable, "-m", "model_converter_tool.cli", "convert",
            "sshleifer/tiny-gpt2", "onnx", "-o", str(cli_output)
        ])
        # Check for model.onnx in the output directory
        assert (cli_output / "model.onnx").exists()

# --- Batch conversion integration ---
def test_batch_conversion(tmp_output):
    """
    Test batch conversion via CLI with a temporary YAML config file.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tmp_output = tmpdir / "integration_outputs"
        tmp_output.mkdir()
        batch_config = tmpdir / "batch.yaml"
        batch_yaml = f"""
tasks:
  - model_path: sshleifer/tiny-gpt2
    output_format: onnx
    output_path: {tmp_output}/batch_gpt2_onnx
"""
        batch_config.write_text(batch_yaml)
        subprocess.check_call([
            sys.executable, "-m", "model_converter_tool.cli", "batch", str(batch_config)
        ])
        # Check for model.onnx in the output directory
        assert (tmp_output / "batch_gpt2_onnx" / "model.onnx").exists()

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