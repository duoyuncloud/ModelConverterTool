# CLI Reference

This document describes the command-line interface (CLI) for the Model Converter Tool, including available commands, options, and usage examples.

## Overview
The CLI allows users to convert, quantize, inspect, and manage machine learning models directly from the terminal.

## Commands
- `convert <input_model> <output_format> [options]`: Convert a model to a different format.
- `batch <config_path> [options]`: Batch convert models using a YAML/JSON config file.
- `inspect <model>`: Inspect and display detailed model information.
- `history`: Show conversion history.
- `config [--action ...] [--key ...] [--value ...]`: Manage tool configuration.

## convert options
- `<input_model>`: Path to the input model file or repo id (required, positional argument)
- `<output_format>`: Target output format (e.g., gguf, onnx, safetensors, etc.) (required, positional argument)
- `-o`, `--output-path`: Path to the output model file or directory (optional, auto-completed if omitted)
- `--quant`: Quantization type (optional, e.g. '4bit', 'q4_k_m', etc.)
- `--quant-config`: Advanced quantization config (optional, JSON string or YAML file)
- `--model-type`: Model type (optional, default: auto)
- `--device`: Device to use (optional, default: auto)
- `--use-large-calibration`: Use large calibration dataset for quantization (optional)
- `--dtype`: Precision for output weights (e.g., fp16, fp32; only for safetensors)
- `--help`: Show help message and exit

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
- `--action`: Action to perform: show/get/set/list_presets (default: show)
- `--key`: Config key (for get/set)
- `--value`: Config value (for set)

## Examples
Convert a model to SafeTensors (fp16):
```bash
python -m model_converter_tool.cli convert path/to/input_model safetensors --dtype fp16 -o path/to/output_model.safetensors
```

Quantize a model:
```bash
python -m model_converter_tool.cli convert path/to/input_model gguf -o path/to/output_model-q4.gguf --quant q4
```

Batch conversion using a config file:
```bash
python -m model_converter_tool.cli batch ../configs/batch_template.yaml
```

Advanced quantization config example:

```bash
python -m model_converter_tool.cli convert path/to/input_model gptq -o path/to/output_model-gptq --quant-config '{"bits":4, "group_size":128, "sym":true, "desc":"custom quant"}'
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
python -m model_converter_tool.cli convert path/to/input_model gptq -o path/to/output_model-gptq --quant-config quant.yaml
```

## Fine-grained Quantization Config (GPTQ/AWQ)

Advanced quantization configuration is fully supported for GPTQ and AWQ engines. You can specify options like `bits`, `group_size`, `sym`, `desc_act`, and more for precise quantization control.

### Supported quantization_config parameters

| Parameter         | Type    | Description                                                                 |
|-------------------|---------|-----------------------------------------------------------------------------|
| bits              | int     | Number of quantization bits (e.g., 4, 8)                                    |
| group_size        | int     | Group size for quantization (e.g., 128)                                     |
| sym               | bool    | Whether to use symmetric quantization                                       |
| desc_act          | bool    | Enable descriptive quantization mechanism (improves accuracy for some models)|
| dynamic           | dict    | Per-layer/module override for quantization params (see advanced usage)      |
| damp_percent      | float   | Damping percent for quantization                                            |
| damp_auto_increment | float | Auto increment for damping                                                  |
| static_groups     | bool    | Use static groups for quantization                                          |
| true_sequential   | bool    | Use true sequential quantization                                            |
| lm_head           | bool    | Quantize the LM head                                                        |
| quant_method      | str     | Quantization method (e.g., 'gptq')                                          |
| format            | str     | Output format (e.g., 'gptq')                                                |
| mse               | float   | MSE loss threshold for quantization                                         |
| parallel_packing  | bool    | Enable parallel packing                                                     |
| meta              | dict    | Extra metadata                                                              |
| device            | str     | Device for quantization ('cpu', 'cuda', etc.)                               |
| pack_dtype        | str     | Data type for packing                                                       |
| adapter           | dict    | Adapter config (for LoRA/EoRA, etc.)                                        |
| rotation          | str     | Rotation type                                                               |
| is_marlin_format  | bool    | Use Marlin kernel format                                                    |

All parameters in the quantization config will be passed to the quantizer for fine-grained control (unsupported keys will be ignored).

## Advanced Usage
- Combining multiple options for custom workflows
- Using different devices (CPU, GPU, Apple Silicon)
- Integrating with shell scripts for automation

## Troubleshooting
- Common errors and solutions
- How to report issues 