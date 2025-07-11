import subprocess
import os
import shutil
import tempfile
import pytest
from model_converter_tool import api

# Paths to example files (adjust if needed)
EXAMPLE_MODEL = "examples/example_basic.py"  # Placeholder for a small model file
BATCH_CONFIG = "configs/batch_template.yaml"

@pytest.fixture(scope="module")
def temp_output_dir():
    dirpath = tempfile.mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath)


def test_cli_convert(temp_output_dir):
    """Test the CLI convert command with a small example model using positional arguments."""
    result = subprocess.run([
        "python3", "-m", "model_converter_tool.cli", "convert",
        EXAMPLE_MODEL,  # input path as positional argument
        temp_output_dir  # output path as positional argument
    ], capture_output=True, text=True)
    assert result.returncode == 0, f"CLI convert failed: {result.stderr}"
    # Check output directory is not empty
    assert os.listdir(temp_output_dir), "Output directory is empty after convert."


def test_api_convert(temp_output_dir):
    """Test the API convert function with a small example model."""
    if not hasattr(api, "convert"):
        pytest.skip("model_converter_tool.api has no 'convert' function. Skipping API convert test.")
    api.convert(EXAMPLE_MODEL, temp_output_dir)
    assert os.listdir(temp_output_dir), "API convert did not produce output."


def test_cli_batch(temp_output_dir):
    """Test the CLI batch command with a sample batch config using positional argument."""
    config_copy = os.path.join(temp_output_dir, "batch_config.yaml")
    shutil.copy(BATCH_CONFIG, config_copy)
    result = subprocess.run([
        "python3", "-m", "model_converter_tool.cli", "batch",
        config_copy  # config path as positional argument
    ], capture_output=True, text=True)
    assert result.returncode == 0, f"CLI batch failed: {result.stderr}"
    assert os.listdir(temp_output_dir), "No batch output generated."


def test_cli_inspect():
    """Test the CLI inspect command on a small example model using positional argument."""
    result = subprocess.run([
        "python3", "-m", "model_converter_tool.cli", "inspect",
        EXAMPLE_MODEL  # model path as positional argument
    ], capture_output=True, text=True)
    assert result.returncode == 0, f"CLI inspect failed: {result.stderr}"
    assert "" in result.stdout or "Model" in result.stdout  # Adjust as needed


def test_cli_config():
    """Test the CLI config command (no extra arguments)."""
    result = subprocess.run([
        "python3", "-m", "model_converter_tool.cli", "config"
    ], capture_output=True, text=True)
    assert result.returncode == 0, f"CLI config failed: {result.stderr}"
    assert result.stdout.strip(), "Config output missing."


def test_invalid_input(temp_output_dir):
    """Test error handling for invalid input file using positional arguments."""
    result = subprocess.run([
        "python3", "-m", "model_converter_tool.cli", "convert",
        "nonexistent.file",  # invalid input path
        temp_output_dir
    ], capture_output=True, text=True)
    assert result.returncode != 0, "CLI convert should fail for nonexistent input."
    assert "error" in result.stderr.lower() or result.stderr.strip(), "No error message for invalid input." 