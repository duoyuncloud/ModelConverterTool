# Model Converter Tool Documentation

Welcome to the documentation for the Model Converter Tool. This guide provides an overview of the tool, installation instructions, usage examples, and technical references for developers and users.

## Overview
A flexible and extensible tool for converting, quantizing, and managing machine learning models across multiple formats and frameworks.

- Supports GGUF, ONNX, HuggingFace, Safetensors, TorchScript, MLX, GPTQ, AWQ, and more.
- muP-to-LLaMA: Automatically detect and rescale muP-initialized models for LLaMA compatibility (`--mup2llama`).
- Fake weight: Generate models with zero or custom-shaped weights for testing (`--fake-weight`, `--fake-weight-config`).
- Fine-grained quantization: Use `--quant` and `--quant-config` for advanced quantization control.

## Installation
Instructions for installing system dependencies and Python packages.

- Run `./install.sh --system-deps` to install system dependencies (macOS, Linux supported).
- Run `./install.sh` to install Python dependencies.
- For Apple Silicon users, add `--enable-mlx` for optimized MLX support.

## Usage
Basic usage examples for converting models:

```bash
python -m model_converter_tool.cli convert gpt2 gguf -o outputs/gpt2.gguf
python -m model_converter_tool.cli convert path/to/mup_model --to safetensors --output path/to/out --mup2llama
python -m model_converter_tool.cli convert gpt2 --to safetensors --output path/to/fake --fake-weight
python -m model_converter_tool.cli batch path/to/batch_config.yaml
python -m model_converter_tool.cli inspect path/to/input_model
python -m model_converter_tool.cli history
python -m model_converter_tool.cli config --action show
```

## CLI Reference
See [cli.md](./cli.md) for all commands and options.

## Configuration
How to use and customize configuration files (YAML/JSON) for batch conversions and advanced workflows. See [config.md](./config.md).

## Conversion Engines
Supported model formats and conversion engines:

- GGUF
- ONNX
- AWQ
- MLX
- GPTQ
- Safetensors
- TorchScript
- HuggingFace (re-save)

## Examples
Links to example scripts and workflows:

- [Basic conversion example](../examples/example_basic.py)
- [Batch conversion example](../examples/example_batch.py)
- [Quantization example](../examples/example_quantization.py)
- [API usage example](../examples/example_api.py)
- [muP-to-LLaMA example](../examples/example_mup2llama.py)
- [Fake weight example](../examples/example_fake_weight.py)

## Contribution
Guidelines for contributing, reporting issues, and submitting pull requests.

- Please see [CONTRIBUTING.md](../CONTRIBUTING.md) for details.

## License
This project is licensed under the Apache 2.0 License. 