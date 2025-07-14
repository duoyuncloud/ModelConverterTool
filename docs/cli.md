# CLI Reference

This document describes the command-line interface (CLI) for the Model Converter Tool, including available commands, options, and usage examples.

## Overview
The CLI allows users to convert, quantize, and manage machine learning models directly from the terminal.

## Command Syntax
```bash
python -m model_converter_tool.cli convert <input_model> --to <format> -o <output_path> [options]
python -m model_converter_tool.cli batch <config_path>
```

## Options
- `<input_model>`: Path to the input model file or repo id (required, positional argument)
- `--to`: Target output format (e.g., gguf, onnx, safetensors, etc.) (required)
- `-o`, `--output-path`: Path to the output model file or directory (optional, auto-completed if omitted)
- `--quant`: Quantization type (optional, e.g. '4bit', 'q4_k_m', etc.)
- `--quant-config`: Advanced quantization config (optional, JSON string or YAML file). Supports keys like `bits`, `group_size`, `sym`, `desc`.
- `--model-type`: Model type (optional)
- `--device`: Device to use (optional)
- `--use-large-calibration`: Use large calibration dataset (optional)
- `--help`: Show help message and exit

## Examples
Convert a model to SafeTensors (fp16):
```bash
python -m model_converter_tool.cli convert path/to/input_model --to safetensors --dtype fp16 -o path/to/output_model.safetensors
```

Quantize a model:
```bash
python -m model_converter_tool.cli convert path/to/input_model --to gguf -o path/to/output_model-q4.gguf --quant q4
```

Batch conversion using a config file:
```bash
python -m model_converter_tool.cli batch ../configs/batch_template.yaml
```

Advanced quantization config example:

```bash
python -m model_converter_tool.cli convert path/to/input_model --to gptq -o path/to/output_model-gptq --quant-config '{"bits":4, "group_size":128, "sym":true, "desc":"custom quant"}'
```

Or using a YAML file:

```yaml
# quant.yaml
bits: 4
group_size: 128
sym: true
desc: my custom quant
```

```bash
python -m model_converter_tool.cli convert path/to/input_model --to gptq -o path/to/output_model-gptq --quant-config quant.yaml
```

## Advanced Usage
- Combining multiple options for custom workflows
- Using different devices (CPU, GPU, Apple Silicon)
- Integrating with shell scripts for automation

## Troubleshooting
- Common errors and solutions
- How to report issues 