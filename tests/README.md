# Test Suite

This directory contains all automated tests for the Model Converter Tool. Each test module targets a specific feature or workflow:

- **test_basic_conversions.py**: Basic model format conversions (ONNX, GGUF, MLX, safetensors, torchscript, HuggingFace, etc).
- **test_quantization.py**: Quantization workflows and config propagation.
- **test_fake_weight.py**: Fake weight generation, including custom shapes and config file support.
- **test_mup2llama.py**: muP-to-LLaMA scaling and config adaptation.
- **test_cli.py**: CLI command coverage, help/version, error handling, and argument validation.
- **test_disk_space.py**: Disk space checks and related edge cases.
- **test_integration.py**: High-coverage integration tests for API, CLI, batch conversion, config/history, and error handling.

## Running Tests

To run all tests:
```bash
pytest
```

For release validation, ensure `test_integration.py` and `test_mup2llama.py` pass, as they cover end-to-end and advanced scenarios.

## CI Integration Test

To manually trigger the integration test in CI:
1. Go to the repository on GitHub.
2. Click the "Actions" tab.
3. Select the "CI" workflow.
4. Click "Run workflow" and choose the branch.
5. Monitor progress and results in the Actions tab. 