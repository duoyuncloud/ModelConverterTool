# Test Suite Documentation

This directory contains comprehensive tests for ModelConverterTool, organized by functionality.

## Test Structure

### Test Categories

- **`test_basic_conversions.py`** - Basic model format conversions using gpt2
  - Tests: gpt2 → onnx, gguf, mlx, fp16, torchscript, safetensors, hf
  - Marker: `@pytest.mark.basic` + `@pytest.mark.fast`

- **`test_quantization.py`** - Quantization format tests using sshleifer/tiny-gpt2
  - Tests: GPTQ, AWQ, GGUF with different quantization levels
  - Marker: `@pytest.mark.quantization` + `@pytest.mark.slow`

- **`test_batch_conversion.py`** - Batch conversion functionality
  - Tests: YAML configuration, dictionary configuration, batch processing
  - Marker: `@pytest.mark.batch` + `@pytest.mark.slow`

- **`test_api_usage.py`** - API usage examples
  - Tests: ModelConverter().convert, ModelConverter().batch_convert
  - Marker: `@pytest.mark.api` + `@pytest.mark.fast`

### Test Markers

- **`fast`** - Quick tests that should run in CI
- **`slow`** - Longer tests that may be skipped in CI
- **`basic`** - Basic conversion functionality
- **`quantization`** - Quantization-specific tests
- **`batch`** - Batch processing tests
- **`api`** - API usage tests

## Running Tests

### Individual Test Files
```bash
# Run specific test file
pytest tests/test_basic_conversions.py -v

# Run with coverage
pytest tests/test_basic_conversions.py --cov=model_converter_tool
```

### Test Categories
```bash
# Run all basic conversion tests
pytest -m basic

# Run all quantization tests
pytest -m quantization

# Run all batch conversion tests
pytest -m batch

# Run all API usage tests
pytest -m api
```

### Speed-based Selection
```bash
# Run only fast tests
pytest -m fast

# Run only slow tests
pytest -m slow

# Run all tests except slow ones
pytest -m "not slow"
```

### Using Test Runner Script
```bash
# Run all tests
python run_tests.py --category all

# Run specific category
python run_tests.py --category basic --verbose

# Run fast tests with coverage
python run_tests.py --fast --coverage
```

## Test Outputs

Test outputs are stored in `test_outputs/` directory:
- `test_outputs/basic_conversions/` - Basic conversion test outputs
- `test_outputs/quantization/` - Quantization test outputs
- `test_outputs/batch_conversion/` - Batch conversion test outputs
- `test_outputs/api_usage/` - API usage test outputs

## Pre-commit Testing

Before committing, run:
```bash
python pre_commit_test.py
```

This will:
1. Run fast tests first (must pass)
2. Run slow tests (warnings only, won't block commit)
3. Provide clear feedback on test status

## Test Models

- **gpt2** - Used for basic conversions (fast, reliable)
- **sshleifer/tiny-gpt2** - Used for quantization tests (smaller, faster)

## Adding New Tests

1. Create test file in `tests/` directory
2. Use appropriate test markers
3. Follow naming convention: `test_*.py`
4. Use class-based tests with descriptive names
5. Include proper setup and teardown
6. Add to appropriate test category

## Test Configuration

See `pytest.ini` for test configuration including:
- Test discovery patterns
- Markers definition
- Default options
- Output formatting 