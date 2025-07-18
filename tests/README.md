# Test Suite for Model Converter Tool

This directory contains all automated tests for the Model Converter Tool. Each module targets a specific feature or workflow:

- **test_basic_conversions.py**: Basic model format conversions (ONNX, GGUF, MLX, safetensors, torchscript, HuggingFace, etc).
- **test_quantization.py**: Quantization workflows and quantization config propagation.
- **test_fake_weight.py**: Fake weight generation, including custom shape and config file support.
- **test_mup2llama.py**: muP (μ-Parametrization) to LLaMA scaling, config adaptation, and correctness of --mup2llama option.
- **test_cli.py**: CLI command coverage, help/version, error handling, and argument validation.
- **test_disk_space.py**: Disk space checks, file size estimation, and related edge cases.
- **test_integration.py**: High-coverage integration tests for API, CLI, batch conversion, config/history, and error handling. Run this before releases.

## Running Tests

To run all tests:
```sh
pytest
```

For release validation, ensure both `test_integration.py` and `test_mup2llama.py` pass, as they cover end-to-end and advanced parameter adaptation scenarios.

## CI/Manual Integration Test

To manually trigger the high-coverage integration test (`test_integration.py`) in CI:
1. Go to the repository on GitHub.
2. Click the "Actions" tab.
3. Select the "Test" workflow.
4. Click "Run workflow" and choose the branch (usually `main`).
5. Monitor progress and results in the Actions tab.

This is recommended before releases or after major changes. 