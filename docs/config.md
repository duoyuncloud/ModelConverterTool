# Configuration Reference

Configuration files enable batch processing and advanced workflows using YAML or JSON format.

## Overview

Configuration files define multiple model conversion tasks with input, output, format, and optional parameters in a single file.

## Basic Structure

```yaml
models:
  - model_path: input/model/path
    output_path: output/model/path
    output_format: target_format
    # optional parameters...
```

## Supported Fields

| Field | Type | Description |
|-------|------|-------------|
| `model_path` | string | Path to input model file or HuggingFace repo ID |
| `output_path` | string | Path for output model file/directory |
| `output_format` | string | Target format (onnx, gguf, safetensors, gptq, awq, mlx, etc.) |
| `quantization` | string | Quantization type (e.g., '4bit', 'q4_k_m') |
| `quantization_config` | dict | Advanced quantization configuration |
| `model_type` | string | Model type (default: auto) |
| `device` | string | Device to use (default: auto) |
| `dtype` | string | Output precision (fp16, fp32; safetensors only) |
| `mup2llama` | boolean | Enable muP-to-LLaMA scaling |
| `fake_weight` | boolean | Use zero weights |
| `fake_weight_shape_dict` | dict | Custom fake weight shapes |
| `use_large_calibration` | boolean | Use large calibration dataset |

## Examples

### Basic Conversion
```yaml
models:
  - model_path: gpt2
    output_path: outputs/gpt2_onnx
    output_format: onnx
  
  - model_path: facebook/opt-125m
    output_path: outputs/opt_safetensors
    output_format: safetensors
    dtype: fp16
```

### Quantization
```yaml
models:
  - model_path: microsoft/DialoGPT-medium
    output_path: outputs/dialogpt_gptq
    output_format: gptq
    quantization: 4bit
  
  - model_path: facebook/opt-1.3b
    output_path: outputs/opt_awq_custom
    output_format: awq
    quantization_config:
      bits: 4
      group_size: 128
      sym: true
      desc_act: true
```

### muP-to-LLaMA Conversion
```yaml
models:
  - model_path: path/to/mup_model
    output_path: outputs/llama_model
    output_format: safetensors
    mup2llama: true
```

### Fake Weights for Testing
```yaml
models:
  - model_path: gpt2
    output_path: outputs/fake_model
    output_format: safetensors
    fake_weight: true
  
  - model_path: gpt2
    output_path: outputs/custom_fake
    output_format: safetensors
    fake_weight: true
    fake_weight_shape_dict:
      embed_tokens.weight: [32000, 4096]
      lm_head.weight: [32000, 4096]
```

## Advanced Quantization

### GPTQ Configuration
```yaml
models:
  - model_path: huggyllama/llama-7b
    output_path: outputs/llama_gptq
    output_format: gptq
    quantization_config:
      bits: 4
      group_size: 128
      sym: true
      desc_act: true
      damp_percent: 0.1
      true_sequential: true
```

### AWQ Configuration
```yaml
models:
  - model_path: microsoft/DialoGPT-large
    output_path: outputs/dialogpt_awq
    output_format: awq
    quantization_config:
      bits: 4
      group_size: 128
      sym: false
      rotation: true
```

## Best Practices

- Use descriptive output paths to avoid overwriting files
- Test configurations with small models first
- Validate YAML/JSON syntax before running batch jobs
- Use `fake_weight: true` for testing workflows without downloading large models
- Enable `use_large_calibration: true` for better quantization quality (slower)

## Running Batch Jobs

```bash
# Run batch conversion
modelconvert batch config.yaml

# With custom settings
modelconvert batch config.yaml --max-workers 2 --max-retries 3
``` 