# CLI Reference

This document describes the command-line interface (CLI) for the Model Converter Tool, including available commands, options, and usage examples.

## Overview
The CLI allows users to convert, quantize, and manage machine learning models directly from the terminal.

## Command Syntax
```bash
python -m model_converter_tool.cli convert <input_model> --to <format> --output <output_path> [options]
python -m model_converter_tool.cli batch <config_path>
```

## Options
- `<input_model>`: Path to the input model file or repo id (required, positional argument)
- `--to`: Target output format (e.g., gguf, onnx, fp16, etc.) (required)
- `--output`: Path to the output model file or directory (optional, auto-completed if omitted)
- `--quant`: Quantization type (optional)
- `--model-type`: Model type (optional)
- `--device`: Device to use (optional)
- `--use-large-calibration`: Use large calibration dataset (optional)
- `--help`: Show help message and exit

## Examples
Convert a model to GGUF format:
```bash
python -m model_converter_tool.cli convert path/to/input_model --to gguf --output path/to/output_model.gguf
```

Quantize a model:
```bash
python -m model_converter_tool.cli convert path/to/input_model --to gguf --output path/to/output_model-q4.gguf --quant q4
```

Batch conversion using a config file:
```bash
python -m model_converter_tool.cli batch ../configs/batch_template.yaml
```

## Advanced Usage
- Combining multiple options for custom workflows
- Using different devices (CPU, GPU, Apple Silicon)
- Integrating with shell scripts for automation

## Troubleshooting
- Common errors and solutions
- How to report issues 