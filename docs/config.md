# Configuration File Reference

This document explains how to use configuration files with the Model Converter Tool for batch conversions and advanced workflows.

## Overview
Configuration files allow you to define multiple model conversion tasks and advanced options in a single YAML or JSON file.

## Configuration Structure
A configuration file typically contains a list of conversion jobs, each specifying input, output, format, and optional parameters.

## Example Config (YAML)
```yaml
models:
  - model_path: path/to/model1.bin
    output_path: path/to/model1.gguf
    output_format: gguf
    quantization: q4
    model_type: auto
    device: auto
  - model_path: path/to/model2.bin
    output_path: path/to/model2.onnx
    output_format: onnx
    model_type: bert
    device: cpu
```

## Supported Fields
- `model_path`: Path to the input model file
- `output_path`: Path to the output model file
- `output_format`: Target output format (e.g., gguf, onnx, fp16, etc.)
- `quantization`: Quantization type (optional)
- `quantization_config`: Advanced quantization config (optional, dict). Supports keys like `bits`, `group_size`, `sym`, `desc`.
- `model_type`: Model type (optional)
- `device`: Device to use (optional)
- `use_large_calibration`: Use large calibration dataset (optional)

## Tips
- Use descriptive output paths to avoid overwriting files
- Validate your config file before running batch conversions

## Advanced Quantization Config Example

```yaml
models:
  - model_path: path/to/model1.bin
    output_path: path/to/model1.gptq
    output_format: gptq
    quantization_config:
      bits: 4
      group_size: 128
      sym: true
      desc: my custom quant
``` 

## Fine-grained Quantization Config (GPTQ/AWQ)

You can use advanced quantization configuration for GPTQ and AWQ engines, supporting options like `bits`, `group_size`, `sym`, `desc`, and more. All parameters in the quantization config will be passed to the quantizer for fine-grained control.

**Example:**

```yaml
quantization_config:
  bits: 4
  group_size: 128
  sym: true
  desc: my custom quant
```

This enables precise control over quantization behavior for your models. 