# Model Converter Tool Documentation

Welcome to the documentation for the Model Converter Tool. This guide provides an overview of the tool, installation instructions, usage examples, and technical references for developers and users.

## Overview
A flexible and extensible tool for converting, quantizing, and managing machine learning models across multiple formats and frameworks.

## Installation
Instructions for installing system dependencies and Python packages.

- Run `./install.sh --system-deps` to install system dependencies (macOS, Linux supported).
- Run `./install.sh` to install Python dependencies.
- For Apple Silicon users, add `--enable-mlx` for optimized MLX support.

## Usage
Basic usage examples for converting models:

```bash
python -m model_converter_tool.cli convert path/to/input_model --to gguf --output path/to/output_model.gguf
```

## CLI Reference
Detailed command-line interface options and flags.

- `<input_model>`: Path to the input model file or repo id (required, positional argument)
- `--to`: Target output format (e.g., gguf, onnx, safetensors, etc.) (required)
- `--output`: Path to the output model file or directory (optional, auto-completed if omitted)
- `--quant`: Quantization type (optional)
- `--model-type`: Model type (optional)
- `--device`: Device to use (optional)

## Configuration
How to use and customize configuration files (YAML/JSON) for batch conversions and advanced workflows.

## Conversion Engines
Supported model formats and conversion engines:

- GGUF
- ONNX
- AWQ
- MLX
- GPTQ
- Safetensors
- TorchScript
- ...and more

## Examples
Links to example scripts and workflows:

- [Basic conversion example](../examples/example_basic.py)
- [Batch conversion example](../examples/example_batch.py)
- [Quantization example](../examples/example_quantization.py)
- [API usage example](../examples/example_api.py)

## Contribution
Guidelines for contributing, reporting issues, and submitting pull requests.

- Please see [CONTRIBUTING.md](../CONTRIBUTING.md) for details.

## License
This project is licensed under the Apache 2.0 License. 