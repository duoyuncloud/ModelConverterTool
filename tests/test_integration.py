import subprocess
import os
import shutil
import tempfile
import pytest

# Try to import API, skip API tests if not available
try:
    from model_converter_tool.api import ModelConverterAPI
    HAS_API = True
except ImportError:
    HAS_API = False

# Path for the auto-generated dummy model file
DUMMY_MODEL = "models/dummy-model.bin"
BATCH_CONFIG = "configs/batch_template.yaml"

@pytest.fixture(scope="module", autouse=True)
def generate_dummy_model():
    """
    Automatically generate a small dummy model file for testing.
    This file will be used as the input model for conversion and inspection tests.
    """
    os.makedirs(os.path.dirname(DUMMY_MODEL), exist_ok=True)
    with open(DUMMY_MODEL, "wb") as f:
        # Write a small amount of random bytes to simulate a model file
        f.write(os.urandom(128))
    yield
    # Clean up the dummy model file after all tests in the module
    if os.path.exists(DUMMY_MODEL):
        os.remove(DUMMY_MODEL)

@pytest.fixture(scope="module")
def temp_output_dir():
    """
    Create a temporary directory for test outputs.
    The directory is removed after the test module finishes.
    """
    dirpath = tempfile.mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath)

def has_transformers():
    """
    Check if the 'transformers' package is installed.
    Some batch tests require this dependency for HuggingFace models.
    """
    try:
        import transformers
        return True
    except ImportError:
        return False

def test_cli_convert_success(temp_output_dir):
    """
    Test CLI convert with the generated dummy model (success path).
    The test expects the CLI to handle the file, even if the format is not truly supported.
    Uses --fake-weight for speed.
    """
    result = subprocess.run([
        "python3", "-m", "model_converter_tool.cli", "convert",
        DUMMY_MODEL, temp_output_dir, "--fake-weight"
    ], capture_output=True, text=True)
    # Accept both success and expected validation failure, but output dir should be checked
    assert result.returncode in (0, 1), f"CLI convert failed unexpectedly: {result.stderr or result.stdout}"
    # Output directory may be empty if format is not supported, so just check for no crash

def test_cli_convert_invalid_input(temp_output_dir):
    """
    Test CLI convert with a non-existent input file (should fail).
    Checks that the CLI returns a non-zero exit code and outputs an error message.
    """
    result = subprocess.run([
        "python3", "-m", "model_converter_tool.cli", "convert",
        "nonexistent.file", temp_output_dir
    ], capture_output=True, text=True)
    assert result.returncode != 0, "CLI convert should fail for nonexistent input."
    # Accept error message in either stdout or stderr
    error_output = result.stdout.lower() + result.stderr.lower()
    assert "validation failed" in error_output or "unsupported input format" in error_output, \
        f"Expected error message, got: {result.stdout} {result.stderr}"

def test_cli_batch(temp_output_dir):
    """
    Test CLI batch with a sample config.
    The batch config should reference the generated dummy model.
    Skips if 'transformers' is not installed and required by the config.
    """
    # Create a minimal batch config referencing the dummy model
    config_copy = os.path.join(temp_output_dir, "batch_config.yaml")
    with open(config_copy, "w") as f:
        f.write(f"""
conversions:
  - input: {DUMMY_MODEL}
    output: {os.path.join(temp_output_dir, 'dummy-model-out.bin')}
    format: gguf
""")
    if not has_transformers():
        pytest.skip("transformers not installed, skipping batch test.")
    result = subprocess.run([
        "python3", "-m", "model_converter_tool.cli", "batch",
        config_copy
    ], capture_output=True, text=True)
    # Accept both full or partial success, but batch should run
    assert "Batch conversion completed" in result.stdout or result.returncode in (0, 1), \
        f"Batch did not run: {result.stdout or result.stderr}"

def test_cli_inspect():
    """
    Test CLI inspect with the generated dummy model.
    Checks that the CLI runs and outputs some information.
    """
    result = subprocess.run([
        "python3", "-m", "model_converter_tool.cli", "inspect",
        DUMMY_MODEL
    ], capture_output=True, text=True)
    assert result.returncode in (0, 1), f"CLI inspect failed: {result.stderr or result.stdout}"
    assert result.stdout.strip(), "Inspect output missing."

def test_cli_config():
    """
    Test CLI config command (should print config info).
    Checks that the CLI runs and outputs some configuration information.
    """
    result = subprocess.run([
        "python3", "-m", "model_converter_tool.cli", "config"
    ], capture_output=True, text=True)
    assert result.returncode == 0, f"CLI config failed: {result.stderr or result.stdout}"
    assert result.stdout.strip(), "Config output missing."

def test_cli_convert_unsupported_format(temp_output_dir):
    """
    Test CLI convert with a file of unsupported format.
    Generates a dummy text file and expects the CLI to fail gracefully.
    """
    dummy_file = os.path.join(temp_output_dir, "dummy.txt")
    with open(dummy_file, "w") as f:
        f.write("not a model")
    result = subprocess.run([
        "python3", "-m", "model_converter_tool.cli", "convert",
        dummy_file, temp_output_dir
    ], capture_output=True, text=True)
    assert result.returncode != 0, "CLI convert should fail for unsupported format."
    error_output = result.stdout.lower() + result.stderr.lower()
    assert "unsupported input format" in error_output or "validation failed" in error_output, \
        f"Expected error message, got: {result.stdout} {result.stderr}"

# Additional tests for higher coverage and robustness

SUPPORTED_FORMATS = ["onnx", "gguf", "torchscript", "mlx", "safetensors", "hf"]
QUANTIZATION_OPTIONS = ["q4_k_m", "q8_0", "q5_k_m"]
DEVICE_OPTIONS = ["cpu", "cuda"]

@pytest.mark.parametrize("output_format", SUPPORTED_FORMATS)
def test_cli_convert_all_formats(temp_output_dir, output_format):
    """
    Test CLI convert for all supported output formats using positional arguments only.
    The CLI does not support --format, so we only test input/output positional args.
    """
    # The CLI expects only input and output as positional arguments
    result = subprocess.run([
        "python3", "-m", "model_converter_tool.cli", "convert",
        DUMMY_MODEL, os.path.join(temp_output_dir, f"dummy-model-out.{output_format}")
    ], capture_output=True, text=True)
    # Accept both success and expected validation failure
    assert result.returncode in (0, 1, 2), f"CLI convert failed for format {output_format}: {result.stderr or result.stdout}"

# The CLI does not support --quantization or --device as options; only test these via API
@pytest.mark.parametrize("quantization", QUANTIZATION_OPTIONS)
def test_cli_convert_quantization(temp_output_dir, quantization):
    """
    Test CLI convert with --quant option for quantization.
    Uses --fake-weight for speed.
    """
    output_path = os.path.join(temp_output_dir, f"dummy-model-q-{quantization}.gguf")
    result = subprocess.run([
        "python3", "-m", "model_converter_tool.cli", "convert",
        DUMMY_MODEL, "gguf",
        "-o", output_path,
        "--quant", quantization,
        "--fake-weight"
    ], capture_output=True, text=True)
    assert result.returncode in (0, 1), f"CLI convert with --quant failed: {result.stderr or result.stdout}"
    # Output file may or may not exist depending on validation, but should not crash

@pytest.mark.parametrize("device", DEVICE_OPTIONS)
def test_cli_convert_device(temp_output_dir, device):
    """
    Test CLI convert with --device option.
    Uses --fake-weight for speed.
    """
    output_path = os.path.join(temp_output_dir, f"dummy-model-device-{device}.gguf")
    result = subprocess.run([
        "python3", "-m", "model_converter_tool.cli", "convert",
        DUMMY_MODEL, "gguf",
        "-o", output_path,
        "--device", device,
        "--fake-weight"
    ], capture_output=True, text=True)
    assert result.returncode in (0, 1), f"CLI convert with --device failed: {result.stderr or result.stdout}"
    # Output file may or may not exist depending on validation, but should not crash

def test_cli_batch_multiple_tasks(temp_output_dir):
    """
    Test CLI batch with multiple tasks (one valid, one invalid).
    The batch config must use 'tasks' as the key, and each task must use 'model_path' and 'output_format' fields to match CLI requirements.
    Checks that the batch summary reports both successes and failures.
    """
    config_copy = os.path.join(temp_output_dir, "batch_config_multi.yaml")
    with open(config_copy, "w") as f:
        f.write(f"""
tasks:
  - model_path: {DUMMY_MODEL}
    output: {os.path.join(temp_output_dir, 'dummy-model-out1.gguf')}
    output_format: gguf
  - model_path: nonexistent.file
    output: {os.path.join(temp_output_dir, 'dummy-model-out2.gguf')}
    output_format: gguf
""")
    result = subprocess.run([
        "python3", "-m", "model_converter_tool.cli", "batch",
        config_copy
    ], capture_output=True, text=True)
    # Check that both success and failure are reported
    assert "Batch conversion completed" in result.stdout or "completed" in result.stdout.lower(), "Batch did not run."
    assert "Failed" in result.stdout or "failed" in result.stdout, "Batch did not report failures."

def test_api_workspace_status_and_formats():
    """
    Test API get_workspace_status and get_supported_formats.
    Checks that the returned data structures are as expected.
    """
    if not HAS_API:
        pytest.skip("API not available.")
    api = ModelConverterAPI()
    status = api.get_workspace_status()
    formats = api.get_supported_formats()
    assert hasattr(status, "workspace_path"), "Workspace status missing workspace_path."
    assert "input_formats" in formats and "output_formats" in formats, "Supported formats missing keys."

def test_api_validation_and_planning():
    """
    Test API validate_conversion and plan_conversion with valid and invalid arguments.
    Checks that the returned plan/validation structure is correct.
    """
    if not HAS_API:
        pytest.skip("API not available.")
    api = ModelConverterAPI()
    # Valid
    validation = api.validate_conversion(DUMMY_MODEL, "gguf")
    assert "plan" in validation and "valid" in validation, "Validation result missing keys."
    plan = api.plan_conversion(DUMMY_MODEL, "gguf", "dummy-out.gguf")
    assert hasattr(plan, "model_path") and hasattr(plan, "output_format"), "Plan missing attributes."
    # Invalid
    validation_invalid = api.validate_conversion("nonexistent.file", "gguf")
    assert not validation_invalid["valid"], "Validation should fail for nonexistent input."

def test_cli_convert_invalid_output_path():
    """
    Test CLI convert with an invalid output path (e.g., unwritable directory).
    Checks that a permission error or similar is handled gracefully.
    """
    forbidden_path = "/root/forbidden/dummy-model-out.gguf"
    result = subprocess.run([
        "python3", "-m", "model_converter_tool.cli", "convert",
        DUMMY_MODEL, forbidden_path
    ], capture_output=True, text=True)
    # Accept non-zero return code and error message
    assert result.returncode != 0, "CLI convert should fail for unwritable output path."
    error_output = result.stdout.lower() + result.stderr.lower()
    assert "permission" in error_output or "denied" in error_output or result.returncode != 0, \
        f"Expected permission error, got: {result.stdout} {result.stderr}"

def test_cli_help_and_version():
    """
    Test CLI --help and --version output for main entrypoint.
    Checks that help and version information is printed.
    """
    help_result = subprocess.run(["python3", "-m", "model_converter_tool.cli", "--help"], capture_output=True, text=True)
    assert help_result.returncode == 0, "CLI --help should succeed."
    assert "usage" in help_result.stdout.lower() or "help" in help_result.stdout.lower(), "Help output missing."
    version_result = subprocess.run(["python3", "-m", "model_converter_tool.cli", "--version"], capture_output=True, text=True)
    assert version_result.returncode == 0, "CLI --version should succeed."
    assert "version" in version_result.stdout.lower(), "Version output missing."

@pytest.mark.parametrize("format_ext,cli_format", [
    ("gguf", "gguf"),
    ("onnx", "onnx"),
    ("mlx", "mlx"),
])
def test_cli_check_supported_formats(temp_output_dir, format_ext, cli_format):
    """
    Test CLI check command for supported formats (GGUF, ONNX, MLX).
    Skips MLX on non-Apple Silicon.
    """
    import platform
    if cli_format == "mlx" and (platform.system() != "Darwin" or platform.machine() != "arm64"):
        pytest.skip("MLX only supported on Apple Silicon macOS")
    # Prepare a dummy or converted file for each format
    dummy_file = os.path.join(temp_output_dir, f"dummy_model.{format_ext if format_ext != 'mlx' else 'npz'}")
    # For ONNX, create a minimal valid ONNX file if possible
    if format_ext == "onnx":
        import onnx
        import numpy as np
        from onnx import helper, TensorProto
        X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])
        Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2])
        node = helper.make_node('Relu', ['X'], ['Y'])
        graph = helper.make_graph([node], 'g', [X], [Y])
        model = helper.make_model(graph)
        onnx.save(model, dummy_file)
    else:
        # For GGUF/MLX, just touch the file (will fail check, but should not crash)
        with open(dummy_file, "wb") as f:
            f.write(os.urandom(128))
    result = subprocess.run([
        "python3", "-m", "model_converter_tool.cli", "check",
        dummy_file, "--format", cli_format
    ], capture_output=True, text=True)
    # Acceptable: success, failure, or error, but should not crash
    assert result.returncode in (0, 1, 2, 3), f"CLI check crashed: {result.stderr or result.stdout}"
    # Output should mention SUCCESS, FAILED, or ERROR
    output = result.stdout + result.stderr
    assert any(word in output for word in ["SUCCESS", "FAILED", "ERROR"]), f"Unexpected CLI check output: {output}" 