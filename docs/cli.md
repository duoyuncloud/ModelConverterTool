# CLI Reference

Complete command-line interface documentation for the Model Converter Tool.

## Overview

The CLI provides commands to convert, quantize, inspect, and manage machine learning models directly from the terminal.

## Commands

### convert
Convert a model to a different format.

**Usage:** `modelconvert convert <input_model> <output_format> [options]`

**Arguments:**
- `<input_model>` - Path to input model file or HuggingFace repo ID
- `<output_format>` - Target format (onnx, gguf, safetensors, gptq, awq, mlx, etc.)

**Options:**
- `-o, --output-path` - Output file/directory path (auto-generated if omitted)
- `--quant` - Quantization type (e.g., '4bit', 'q4_k_m')
- `--quant-config` - Advanced quantization config (JSON string or YAML file)
- `--model-type` - Model type (default: auto)
- `--device` - Device to use (default: auto)
- `--dtype` - Precision for output weights (fp16, fp32; for safetensors only)
- `--mup2llama` - Enable muP-to-LLaMA parameter scaling
- `--fake-weight` - Use zero weights for all parameters
- `--fake-weight-config` - Custom shapes for fake weights (JSON/YAML file)
- `--use-large-calibration` - Use large calibration dataset for quantization

**Examples:**
```bash
# Basic conversion
modelconvert convert gpt2 onnx

# With quantization
modelconvert convert facebook/opt-125m gptq --quant 4bit

# muP to LLaMA conversion
modelconvert convert path/to/mup_model safetensors --mup2llama

# Fake weights for testing
modelconvert convert gpt2 safetensors --fake-weight

# Custom output path
modelconvert convert gpt2 onnx -o models/gpt2_onnx
```

### batch
Batch convert models using a configuration file.

**Usage:** `modelconvert batch <config_path> [options]`

**Arguments:**
- `<config_path>` - Path to batch configuration file (YAML/JSON)

**Options:**
- `--max-workers` - Maximum concurrent workers (default: 1)
- `--max-retries` - Maximum retries per task (default: 1)
- `--skip-disk-check` - Skip disk space checking (not recommended)

### inspect
Display detailed model information.

**Usage:** `modelconvert inspect <model>`

**Arguments:**
- `<model>` - Model path or HuggingFace repo ID

### check
Test if a model can be loaded and run inference.

**Usage:** `modelconvert check <model_path> [options]`

**Arguments:**
- `<model_path>` - Path to model file or directory

**Options:**
- `-f, --format` - Model format (auto-detected if omitted)
- `-v, --verbose` - Show detailed error information

### history
Show conversion history.

**Usage:** `modelconvert history`

### config
Manage configuration settings.

**Usage:** `modelconvert config <subcommand> [args]`

**Subcommands:**
- `show` - Display all configuration values
- `get <key>` - Get specific configuration value
- `set <key> <value>` - Set configuration value
- `list-presets` - List available configuration presets

## Advanced Features

### Quantization Configuration
For GPTQ and AWQ engines, use advanced quantization options:

```bash
# Using quantization config file
modelconvert convert model gptq --quant-config quant_config.yaml

# Inline JSON config
modelconvert convert model gptq --quant-config '{"bits": 4, "group_size": 128, "sym": true}'
```

### muP-to-LLaMA Scaling
Automatically convert muP-initialized models to LLaMA-compatible format:

```bash
modelconvert convert mup_model safetensors --mup2llama
```

### Fake Weights
Generate models with zero or custom-shaped weights for testing:

```bash
# Zero weights
modelconvert convert gpt2 safetensors --fake-weight

# Custom shapes
modelconvert convert gpt2 safetensors --fake-weight --fake-weight-config shapes.yaml
```

## Supported Formats

**Input:** HuggingFace, Safetensors, TorchScript, ONNX, GGUF, MLX
**Output:** HuggingFace, Safetensors, TorchScript, ONNX, GGUF, MLX, GPTQ, AWQ

## Troubleshooting

- Use `--verbose` with the `check` command for detailed error information
- Check conversion history with `modelconvert history` to review past operations
- Ensure sufficient disk space before large conversions
- For format-specific issues, refer to the engine documentation 