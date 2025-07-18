# CLI Reference

This document describes the command-line interface (CLI) for the Model Converter Tool, including available commands, options, and usage examples.

## Overview
The CLI allows users to convert, quantize, inspect, and manage machine learning models directly from the terminal.

## Commands
- `convert <input_model> <output_format> [options]`: Convert a model to a different format.
- `batch <config_path> [options]`: Batch convert models using a YAML/JSON config file.
- `inspect <model>`: Inspect and display detailed model information.
- `history`: Show conversion history.
- `config show`: Show all configuration values.
- `config get <key>`: Get a configuration value by key.
- `config set <key> <value>`: Set a configuration value.
- `config list-presets`: List all available configuration presets.
- `check <model_path> [--format <format>] [--verbose]`: Check if a model file is usable (can be loaded and run a simple inference).

## convert options
- `<input_model>`: Path to the input model file or repo id (required, positional argument)
- `<output_format>`: Target output format (e.g., gguf, onnx, safetensors, etc.) (required, positional argument)
- `-o`, `--output-path`: Path to the output model file or directory (optional, auto-completed if omitted)
- `--quant`: Quantization type (optional, e.g. '4bit', 'q4_k_m', etc.)
- `--quant-config`: Advanced quantization config (optional, JSON string or YAML file)
- `--mup2llama`: Enable muP-to-LLaMA parameter scaling and config adaptation (optional)
- `--fake-weight`: Use zero weights for all parameters (for testing/debugging, optional)
- `--fake-weight-config`: Path to a JSON/YAML file specifying custom shapes for fake weights (optional)
- `--model-type`: Model type (optional, default: auto)
- `--device`: Device to use (optional, default: auto)
- `--use-large-calibration`: Use large calibration dataset for quantization (optional)
- `--dtype`: Precision for output weights (e.g., fp16, fp32; only for safetensors)
- `--help`: Show help message and exit

## Examples
```bash
python -m model_converter_tool.cli convert gpt2 --to gguf --output outputs/gpt2.gguf
python -m model_converter_tool.cli convert path/to/mup_model --to safetensors --output outputs/mup2llama --mup2llama
python -m model_converter_tool.cli convert gpt2 --to safetensors --output outputs/fake --fake-weight
python -m model_converter_tool.cli convert gpt2 --to safetensors --output outputs/fake_custom --fake-weight --fake-weight-config configs/fake_weight.yaml
```

## batch options
- `<config_path>`: Path to the batch configuration file (YAML/JSON)
- `--max-workers`: Maximum number of concurrent workers
- `--max-retries`: Maximum number of retries per task
- `--skip-disk-check`: Skip disk space checking (not recommended)

## inspect options
- `<model>`: Model path or repo id (required)

## history options
- (No arguments)

## config options
Subcommands:
- `show`: Show all configuration values.
- `get <key>`: Get a configuration value by key.
- `set <key> <value>`: Set a configuration value.
- `list-presets`: List all available configuration presets.

## check options
- `<model_path>`: Path to the model file or directory (required)
- `--format`, `-f`: Model format (optional, auto-detected if omitted)
- `--verbose`, `-v`: Show detailed error information (optional)
- `--help`: Show help message and exit

Checks if a model file is usable (i.e., can be loaded and run a simple inference). This is more than just format validation: it attempts to load the model and run a minimal inference to ensure usability. Supports all major formats (GGUF, ONNX, MLX, GPTQ, AWQ, SafeTensors, TorchScript, HuggingFace, etc.).

## Fine-grained Quantization Config (GPTQ/AWQ)

Advanced quantization configuration is fully supported for GPTQ and AWQ engines. You can specify options like `bits`, `group_size`, `sym`, `desc_act`, and more for precise quantization control.

## Advanced Usage
- Combine options for custom workflows (e.g., quantization + muP scaling + fake weight)
- Use different devices (CPU, GPU, Apple Silicon)
- Integrate with shell scripts for automation

## Troubleshooting
- Common errors and solutions
- How to report issues 