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
- `--model-type` - Model type for Megatron conversions (minicpm, llama) or auto-detection
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

# HuggingFace to Megatron-LM conversion
modelconvert convert OpenBMB/MiniCPM4-0.5B hf2megatron --model-type minicpm

# Megatron-LM to HuggingFace conversion
modelconvert convert models/megatron_model hf --model-type minicpm

# MTK conversion for MediaTek platforms
modelconvert convert OpenBMB/MiniCPM4-0.5B mtk --model-type text-generation

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

### Megatron-LM Integration
Bidirectional conversion between HuggingFace and Megatron-LM formats:

```bash
# HuggingFace to Megatron-LM
modelconvert convert OpenBMB/MiniCPM4-0.5B hf2megatron --model-type minicpm

# Megatron-LM to HuggingFace
modelconvert convert models/megatron_model hf --model-type minicpm

# Llama models
modelconvert convert meta-llama/Llama-2-7b hf2megatron --model-type llama
```

**Features:**
- **Automatic format detection**: Detects Megatron-LM format by `model.pt` + `metadata.json`
- **Model structure mapping**: Handles architecture differences between formats
- **Weight reordering**: Properly reorders attention and MLP weights
- **Configuration preservation**: Maintains model metadata across conversions

### Fake Weights
Generate models with zero or custom-shaped weights for testing:

```bash
# Zero weights
modelconvert convert gpt2 safetensors --fake-weight

# Custom shapes
modelconvert convert gpt2 safetensors --fake-weight --fake-weight-config shapes.yaml
```

## Supported Formats

**Input:** HuggingFace, Safetensors, TorchScript, ONNX, GGUF, MLX, **Megatron-LM**
**Output:** HuggingFace, Safetensors, TorchScript, ONNX, GGUF, MLX, GPTQ, AWQ, **Megatron-LM**, **MTK**

### Megatron-LM Support

**Conversion Directions:**
- **HF → Megatron-LM**: `hf2megatron` format
- **Megatron-LM → HF**: `hf` or `huggingface` format

**Supported Models:**
- **MiniCPM**: `--model-type minicpm`
- **Llama/Llama2/Mistral**: `--model-type llama`

**Examples:**
```bash
# HF to Megatron-LM
modelconvert convert OpenBMB/MiniCPM4-0.5B hf2megatron --model-type minicpm

# Megatron-LM to HF
modelconvert convert models/megatron_model hf --model-type minicpm
```

### MTK Support

Convert HuggingFace models to MTK format for MediaTek platforms:

**Supported Platforms:**
- **MT6991**: High-end MediaTek platform
- **MT6989**: Mid-range MediaTek platform  
- **MT6897**: Entry-level MediaTek platform

**Supported Model Sizes:**
- **0_5B, 0_9B, 1_2B, 1_6B, 8B, 0_58B**

**Configuration Options:**
- `--model-type`: `text-generation` (LLM) or `image-classification` (VLM)
- `--quantization-config`: JSON configuration for platform, model size, and other parameters

**Examples:**
```bash
# Basic LLM conversion
modelconvert convert OpenBMB/MiniCPM4-0.5B mtk --model-type text-generation

# VLM conversion with custom platform
modelconvert convert vision-model mtk --model-type image-classification \
  --quantization-config '{"platform": "MT6897", "model_size": "1_6B"}'

# Advanced configuration
modelconvert convert model mtk --model-type text-generation \
  --quantization-config '{
    "platform": "MT6991",
    "model_size": "1_2B", 
    "weight_bit": 4,
    "mtk_cloud_path": "/custom/path/to/mtk_cloud"
  }'
```

**Features:**
- **Automatic model type detection**: Detects LLM vs VLM based on model configuration
- **Real-time progress display**: Shows conversion progress with detailed logging
- **Flexible path configuration**: Supports custom MTK cloud paths via parameters or environment variables
- **Output validation**: Validates conversion results by checking for TFLite files

**Dependencies:**
- Requires separate `mtk_cloud` repository with conversion scripts
- Repository provides `install.sh` for dependency installation
- Uses `run_example_llm_cybertron.sh` for LLM models
- Uses `run_example_vlm_cybertron.sh` for VLM models

## Troubleshooting

- Use `--verbose` with the `check` command for detailed error information
- Check conversion history with `modelconvert history` to review past operations
- Ensure sufficient disk space before large conversions
- For format-specific issues, refer to the engine documentation 