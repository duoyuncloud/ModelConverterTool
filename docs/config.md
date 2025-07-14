# Configuration File Reference

This document explains how to use configuration files with the Model Converter Tool for batch conversions and advanced workflows.

## Overview
Configuration files allow you to define multiple model conversion tasks and advanced options in a single YAML or JSON file.

## Configuration Structure
A configuration file typically contains a list of conversion jobs, each specifying input, output, format, and optional parameters.

## Example Config (YAML)
```yaml
conversions:
  - input: path/to/model1.bin
    output: path/to/model1.gguf
    to: gguf
    quant: q4
    model_type: auto
    device: auto
  - input: path/to/model2.bin
    output: path/to/model2.onnx
    to: onnx
    model_type: bert
    device: cpu
```

## Supported Fields
- `input`: Path to the input model file
- `output`: Path to the output model file
- `to`: Target output format (e.g., gguf, onnx, fp16, etc.)
- `quant`: Quantization type (optional)
- `quantization_config`: Advanced quantization config (optional, dict). Supports keys like `bits`, `group_size`, `sym`, `desc`.
- `model_type`: Model type (optional)
- `device`: Device to use (optional)
- `use_large_calibration`: Use large calibration dataset (optional)

## Tips
- Use descriptive output paths to avoid overwriting files
- Validate your config file before running batch conversions
- Refer to the [examples/batched/README.md](../examples/batched/README.md) for more usage examples 

## Advanced Quantization Config Example

```yaml
conversions:
  - input: path/to/model1.bin
    output: path/to/model1.gptq
    to: gptq
    quantization_config:
      bits: 4
      group_size: 128
      sym: true
      desc: my custom quant
``` 