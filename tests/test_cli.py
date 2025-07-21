import subprocess
import sys
import pytest

CLI_CMD = [sys.executable, "-m", "model_converter_tool.cli"]


@pytest.fixture(scope="module")
def tmp_output(tmp_path_factory):
    d = tmp_path_factory.mktemp("cli_outputs")
    return d


def test_cli_convert_success(tmp_output):
    """
    Test CLI convert command with a simple model and format.
    """
    output_path = tmp_output / "gpt2.onnx"
    result = subprocess.run(
        CLI_CMD + ["convert", "gpt2", "onnx", "-o", str(output_path)], capture_output=True, text=True
    )
    assert result.returncode == 0, f"CLI convert failed: {result.stderr or result.stdout}"
    assert "Conversion succeeded" in result.stdout

    # The output is now a directory (e.g., gpt2_onnx)
    output_dir = tmp_output / "gpt2_onnx"
    assert output_dir.exists() and output_dir.is_dir(), f"Output directory does not exist: {output_dir}"

    # For ONNX, check for model.onnx
    model_file = output_dir / "model.onnx"
    assert model_file.exists(), f"ONNX model file not found: {model_file}"


def test_cli_convert_help():
    """
    Test CLI convert --help output.
    """
    result = subprocess.run(CLI_CMD + ["convert", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "usage" in result.stdout.lower() or "help" in result.stdout.lower()


def test_cli_convert_invalid_format(tmp_output):
    """
    Test CLI convert with unsupported format.
    """
    output_path = tmp_output / "gpt2.invalid"
    result = subprocess.run(
        CLI_CMD + ["convert", "gpt2", "invalidformat", "-o", str(output_path)], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert (
        "unsupported output format" in result.stdout.lower()
        or "validation failed" in result.stdout.lower()
        or "failed" in result.stdout.lower()
    )


def test_cli_inspect():
    """
    Test CLI inspect command.
    """
    result = subprocess.run(CLI_CMD + ["inspect", "gpt2"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Format:" in result.stdout


def test_cli_check():
    """
    Test CLI check command.
    """
    result = subprocess.run(CLI_CMD + ["check", "gpt2"], capture_output=True, text=True)
    assert result.returncode in (0, 2)  # 0: success, 2: failed check
    assert "Model usability check" in result.stdout


def test_cli_config_show():
    """
    Test CLI config show command.
    """
    result = subprocess.run(CLI_CMD + ["config", "show"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "cache_dir" in result.stdout or "config" in result.stdout.lower()


def test_cli_history():
    """
    Test CLI history command.
    """
    result = subprocess.run(CLI_CMD + ["history"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Completed tasks" in result.stdout


def test_cli_main_help():
    """
    Test CLI main --help output.
    """
    result = subprocess.run(CLI_CMD + ["--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "usage" in result.stdout.lower() or "help" in result.stdout.lower()


def test_cli_main_version():
    """
    Test CLI --version output.
    """
    result = subprocess.run(CLI_CMD + ["--version"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "version" in result.stdout.lower()
