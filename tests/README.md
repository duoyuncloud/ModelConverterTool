# Tests

This directory contains all automated tests for the Model Converter Tool.

- **test_basic_conversions.py**: Unit tests for basic model conversions.
- **test_quantization.py**: Unit tests for quantization workflows.
- **test_cli.py**: Unit tests for CLI commands and argument handling.
- **test_disk_space.py**: Tests for disk space and file handling edge cases.
- **test_integration.py**: High-coverage integration tests covering API, CLI, batch conversion, config/history, and error handling. Run this before releases.

## Running Tests

To run all tests:
```sh
pytest
```

For release validation, ensure `test_integration.py` passes. 

## Manually Running Integration Test in CI

To manually trigger the high-coverage integration test (`test_integration.py`) in GitHub Actions:

1. Go to the repository on GitHub.
2. Click the "Actions" tab at the top.
3. Select the "Test" workflow from the left sidebar.
4. Click the "Run workflow" button (top right).
5. Choose the branch (usually `main`) and click the green "Run workflow" button.
6. The integration test job will run and its progress and results can be monitored in the Actions tab.

This is useful for release validation or after major changes. 