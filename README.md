# Model Converter Tool

A professional, API-first tool for machine learning model conversion and management.
Supports ONNX, GGUF, MLX, TorchScript, GPTQ, AWQ, SafeTensors, HuggingFace, **Megatron-LM**, **MTK**, and more.

## Features

- **Multi-format support**: Convert between ONNX, GGUF, MLX, GPTQ, AWQ, SafeTensors, **Megatron-LM**, **MTK**, and more
- **Megatron-LM integration**: Bidirectional conversion between HuggingFace and Megatron-LM formats
- **MTK integration**: Convert HuggingFace models to MTK format for MediaTek platforms
- **Advanced quantization**: Fine-grained control with GPTQ/AWQ configuration
- **muP-to-LLaMA scaling**: Automatic parameter rescaling for LLaMA compatibility
- **Fake weights**: Generate test models without downloading large parameters
- **Batch processing**: Convert multiple models using YAML/JSON configuration
- **API-first design**: Use via CLI or integrate into Python workflows

## Installation

```bash
git clone https://github.com/duoyuncloud/ModelConverterTool.git
cd ModelConverterTool
chmod +x install.sh
./install.sh
source venv/bin/activate
```

## Quick Start

```bash
# Convert a single model
modelconvert convert gpt2 onnx

# Convert with quantization
modelconvert convert facebook/opt-125m gptq --quant 4bit

# Convert muP model to LLaMA format
modelconvert convert path/to/mup_model safetensors --mup2llama

# Convert HuggingFace to Megatron-LM format
modelconvert convert OpenBMB/MiniCPM4-0.5B hf2megatron --model-type minicpm

# Convert Megatron-LM to HuggingFace format
modelconvert convert models/megatron_model hf --model-type minicpm

# Convert to MTK format for MediaTek platforms
modelconvert convert OpenBMB/MiniCPM4-0.5B mtk --model-type text-generation

# Generate fake weights for testing
modelconvert convert gpt2 safetensors --fake-weight

# Batch convert multiple models
modelconvert batch configs/batch_config.yaml

# Inspect model details
modelconvert inspect gpt2

# Check model usability
modelconvert check outputs/model.onnx

# View conversion history
modelconvert history
```

## Commands

### convert
Convert a model to a different format.

**Usage:** `modelconvert convert <input_model> <output_format> [options]`

**Options:**
- `-o, --output-path` - Output file/directory path
- `--quant` - Quantization type (4bit, q4_k_m, etc.)
- `--quant-config` - Advanced quantization config (JSON/YAML)
- `--mup2llama` - Enable muP-to-LLaMA scaling
- `--model-type` - Model type for Megatron conversions (minicpm, llama)
- `--fake-weight` - Use zero weights for testing
- `--dtype` - Output precision (fp16, fp32)

### batch
Batch convert models using a configuration file.

**Usage:** `modelconvert batch <config_path> [options]`

**Options:**
- `--max-workers` - Concurrent workers (default: 1)
- `--max-retries` - Max retries per task (default: 1)

### inspect
Display detailed model information.

**Usage:** `modelconvert inspect <model>`

### check
Test if a model can be loaded and run inference.

**Usage:** `modelconvert check <model_path> [--format <format>] [--verbose]`

### history
Show conversion history.

**Usage:** `modelconvert history`

### config
Manage configuration settings.

**Usage:** `modelconvert config <show|get|set|list-presets> [args]`

## Advanced Features

### Quantization Configuration
Fine-grained control for GPTQ and AWQ engines:

```bash
# Using config file
modelconvert convert model gptq --quant-config config.yaml

# Inline JSON
modelconvert convert model gptq --quant-config '{"bits": 4, "group_size": 128}'
```

**Supported Parameters:**
- `bits` - Quantization bits (4, 8)
- `group_size` - Group size (128, 256)
- `sym` - Symmetric quantization (bool)
- `desc_act` - Descriptive activation (bool)
- `damp_percent` - Damping percentage
- And more...

### muP-to-LLaMA Scaling
Automatically convert muP-initialized models:

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

**Supported Models:**
- **MiniCPM**: Full bidirectional support
- **Llama/Llama2**: Full bidirectional support
- **Mistral**: Full bidirectional support

### MTK Integration
Convert HuggingFace models to MTK format for MediaTek platforms:

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

**Supported Platforms:**
- **MT6991**: High-end MediaTek platform
- **MT6989**: Mid-range MediaTek platform  
- **MT6897**: Entry-level MediaTek platform

**Supported Model Sizes:**
- **0_5B, 0_9B, 1_2B, 1_6B, 8B, 0_58B**

**Features:**
- Automatic model type detection (LLM vs VLM)
- Real-time conversion progress display
- Custom MTK cloud path configuration
- Environment variable support (`MTK_CLOUD_PATH`)
- Output validation with TFLite file detection

**Dependencies:**
- MTK conversion requires a separate `mtk_cloud` repository with `install.sh`
- The repository provides conversion scripts for LLM and VLM models

### Fake Weights
Generate models with zero weights for testing:

```bash
# Zero weights
modelconvert convert gpt2 safetensors --fake-weight

# Custom shapes
modelconvert convert gpt2 safetensors --fake-weight --fake-weight-config shapes.yaml
```

## Supported Formats

### Conversion Matrix

| Input Format | Output Formats |
|--------------|----------------|
| HuggingFace  | All formats |
| SafeTensors  | HuggingFace, SafeTensors |
| TorchScript  | TorchScript |
| ONNX         | ONNX |
| GGUF         | GGUF |
| MLX          | MLX |
| Megatron-LM  | HuggingFace |
| **MTK**      | **MTK** |

### Quantization Support

| Format | Quantization Types |
|--------|--------------------|
| GPTQ   | 4bit, 8bit |
| AWQ    | 4bit, 8bit |
| GGUF   | q4_k_m, q5_k_m, q8_0 |
| MLX    | q4_k_m, q8_0, q5_k_m |
| SafeTensors | fp16, fp32 |
| **MTK** | **4bit, 8bit** |

## API Usage

```python
from model_converter_tool.api import ModelConverterAPI

api = ModelConverterAPI()

# Convert with fake weights for testing
result = api.convert_model(
    model_name="gpt2",
    output_format="onnx",
    output_path="./gpt2.onnx",
    fake_weight=True
)
```

## Configuration Files

Create YAML/JSON files for batch processing:

```yaml
models:
  - model_path: gpt2
    output_path: outputs/gpt2_onnx
    output_format: onnx
  
  - model_path: facebook/opt-125m
    output_path: outputs/opt_gptq
    output_format: gptq
    quantization: 4bit
```

## Testing

Run the test suite:

```bash
# All tests
pytest

# Integration tests
pytest tests/test_integration.py

# Specific test
pytest tests/test_cli.py
```

## Documentation

- **[CLI Reference](docs/cli.md)** - Complete command documentation
- **[Configuration](docs/config.md)** - Batch processing and advanced options
- **[Converter Engine](docs/converter.md)** - Technical details and architecture
- **[Megatron-LM Integration](docs/megatron.md)** - Comprehensive guide for Megatron-LM conversions
- **[Examples](examples/)** - Sample scripts and workflows