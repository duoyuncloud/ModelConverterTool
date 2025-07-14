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
python -m model_converter_tool.cli convert path/to/input_model gguf -o path/to/output_model.gguf
python -m model_converter_tool.cli batch path/to/batch_config.yaml
python -m model_converter_tool.cli inspect path/to/input_model
python -m model_converter_tool.cli history
python -m model_converter_tool.cli config --action show
```

## CLI Reference
Detailed command-line interface options and flags.

### Commands
- `convert <input_model> <output_format>`: Convert a model to a different format.
- `batch <config_path>`: Batch convert models using a YAML/JSON config file.
- `inspect <model>`: Inspect and display detailed model information.
- `history`: Show conversion history.
- `config [--action ...] [--key ...] [--value ...]`: Manage global/local configuration.

### convert options
- `<input_model>`: Path to the input model file or repo id (required, positional argument)
- `<output_format>`: Target output format (e.g., gguf, onnx, safetensors, etc.) (required, positional argument)
- `-o`, `--output-path`: Path to the output model file or directory (optional, auto-completed if omitted)
- `--quant`: Quantization type (optional)
- `--quant-config`: Advanced quantization config (optional, JSON string or YAML file)
- `--model-type`: Model type (optional, default: auto)
- `--device`: Device to use (optional, default: auto)
- `--use-large-calibration`: Use large calibration dataset for quantization (optional)
- `--dtype`: Precision for output weights (e.g., fp16, fp32; only for safetensors)
- `--help`: Show help message and exit

### batch options
- `<config_path>`: Path to the batch configuration file (YAML/JSON)
- `--max-workers`: Maximum number of concurrent workers
- `--max-retries`: Maximum number of retries per task
- `--skip-disk-check`: Skip disk space checking (not recommended)

### inspect options
- `<model>`: Model path or repo id (required)

### history options
- (No arguments)

### config options
- `--action`: Action to perform: show/get/set/list_presets (default: show)
- `--key`: Config key (for get/set)
- `--value`: Config value (for set)

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