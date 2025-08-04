# Model Converter Engine

Technical documentation for the model conversion engine, supported formats, and internal architecture.

## Overview

The conversion engine transforms machine learning models between different formats, applies quantization, and optimizes for various hardware backends using an API-first architecture.

## Architecture

```
CLI → Core → API → Converter → Engine Modules
```

- **CLI**: Handles user input and parameters
- **Core**: Business logic using API calls
- **API**: Central interface for all operations
- **Converter**: Dispatches to appropriate engine modules
- **Engine Modules**: Format-specific conversion logic

## Supported Formats

### Input Formats
- HuggingFace (transformers models)
- Safetensors
- TorchScript
- ONNX
- GGUF
- MLX
- Megatron-LM

### Output Formats
- HuggingFace
- Safetensors
- TorchScript
- ONNX
- GGUF
- MLX
- GPTQ (quantized)
- AWQ (quantized)
- Megatron-LM

### Conversion Matrix

| Input Format | Supported Output Formats |
|--------------|-------------------------|
| HuggingFace  | All formats |
| Safetensors  | HuggingFace, Safetensors |
| TorchScript  | TorchScript |
| ONNX         | ONNX |
| GGUF         | GGUF |
| MLX          | MLX |
| Megatron-LM  | HuggingFace |
| **MTK**      | **MTK** |

> **Note:** Some formats (RK, AX, QNN) are planned for future releases.

## Features

### muP-to-LLaMA Scaling
Automatically detects and converts muP-initialized models to LLaMA-compatible format:
- Rescales parameters according to muP → LLaMA conversion rules
- Updates model configuration
- Preserves model functionality

### Fake Weights
Generates models with zero or custom-shaped weights for testing:
- Zero weights: All parameters set to zero
- Custom shapes: User-defined parameter shapes
- Useful for testing workflows without large model downloads

### Advanced Quantization
Fine-grained quantization control for GPTQ and AWQ:
- Configurable bits, group size, symmetry
- Custom calibration datasets
- Hardware-optimized formats

### Megatron-LM Integration
Bidirectional conversion between HuggingFace and Megatron-LM formats:

**Supported Models:**
- **MiniCPM**: Full bidirectional support with optimized weight mapping
- **Llama/Llama2**: Complete conversion with attention and MLP layer mapping
- **Mistral**: Full support for Mistral architecture variants

**Conversion Features:**
- **Automatic format detection**: Detects Megatron-LM format by presence of `model.pt` + `metadata.json`
- **Model structure mapping**: Handles differences between HuggingFace and Megatron-LM architectures
- **Weight reordering**: Properly reorders attention and MLP weights for format compatibility
- **Configuration preservation**: Maintains model configuration and metadata across conversions

**Usage Examples:**
```bash
# HuggingFace to Megatron-LM
modelconvert convert OpenBMB/MiniCPM4-0.5B hf2megatron --model-type minicpm

# Megatron-LM to HuggingFace  
modelconvert convert models/megatron_model hf --model-type minicpm

# Llama models
modelconvert convert meta-llama/Llama-2-7b hf2megatron --model-type llama
```

### MTK Integration
Convert HuggingFace models to MTK format for MediaTek platforms:

**Supported Platforms:**
- **MT6991**: High-end MediaTek platform
- **MT6989**: Mid-range MediaTek platform  
- **MT6897**: Entry-level MediaTek platform

**Supported Model Sizes:**
- **0_5B, 0_9B, 1_2B, 1_6B, 8B, 0_58B**

**Conversion Features:**
- **Automatic model type detection**: Detects LLM vs VLM based on model configuration
- **Real-time progress display**: Shows conversion progress with detailed logging
- **Flexible path configuration**: Supports custom MTK cloud paths via parameters or environment variables
- **Output validation**: Validates conversion results by checking for TFLite files
- **Platform-specific optimization**: Optimizes for different MediaTek platforms

**Usage Examples:**
```bash
# Basic LLM conversion
modelconvert convert OpenBMB/MiniCPM4-0.5B mtk --model-type text-generation

# VLM conversion with custom platform
modelconvert convert vision-model mtk --model-type image-classification \
  --quantization-config '{"platform": "MT6897", "model_size": "1_6B"}'

# Advanced configuration with custom path
modelconvert convert model mtk --model-type text-generation \
  --quantization-config '{
    "platform": "MT6991",
    "model_size": "1_2B", 
    "weight_bit": 4,
    "mtk_cloud_path": "/custom/path/to/mtk_cloud"
  }'
```

**Dependencies:**
- Requires separate `mtk_cloud` repository with conversion scripts
- Repository provides `install.sh` for dependency installation
- Uses `run_example_llm_cybertron.sh` for LLM models
- Uses `run_example_vlm_cybertron.sh` for VLM models

## Engine Modules

Each engine module exposes three core functions:

### convert(model, tokenizer, model_name, output_path, ...)
Converts the model to the target format.

**Returns:** `(success: bool, extra_info: dict or None)`

### validate(path, *args, **kwargs)
Validates that the converted model file is properly formatted.

**Returns:** `bool` - True if valid, False otherwise

### can_infer(path, *args, **kwargs)
Tests if the model can be loaded and run inference.

**Returns:** `bool` - True if inference works, False otherwise

## Conversion Workflow

1. **Load Input Model**: Parse and load the source model
2. **Apply Transformations**: 
   - muP-to-LLaMA scaling (if enabled)
   - Fake weight generation (if enabled)
3. **Quantization**: Apply quantization configuration (if specified)
4. **Format Conversion**: Convert to target format using appropriate engine
5. **Validation**: Verify output model integrity
6. **Save Output**: Write converted model to specified path

## Device Support

- **CPU**: All formats supported
- **CUDA GPU**: Optimized for quantization and large models
- **Apple Silicon (MPS)**: MLX format with hardware acceleration
- **Auto-detection**: Automatically selects optimal device

## Extension Guide

To add support for a new format:

1. Create a new engine module in `model_converter_tool/engine/`
2. Implement the three required functions: `convert`, `validate`, `can_infer`
3. Follow the established function signatures and return types
4. Add format detection logic to the converter
5. Update documentation and tests

## Error Handling

- **Graceful Degradation**: Fallback to CPU if GPU unavailable
- **Detailed Logging**: Comprehensive error messages and stack traces
- **Validation**: Pre-conversion checks for compatibility
- **Recovery**: Automatic retry mechanisms for transient failures

## Performance Considerations

- **Lazy Loading**: Engine modules loaded on-demand
- **Memory Management**: Efficient model loading and cleanup
- **Parallel Processing**: Batch conversion with configurable workers
- **Disk Space**: Pre-conversion space checks

## References

- [API Documentation](../model_converter_tool/api.py)
- [Engine Source Code](../model_converter_tool/engine/)
- [Test Suite](../tests/)
- [Example Scripts](../examples/) 