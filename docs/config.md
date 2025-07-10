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
- `model_type`: Model type (optional)
- `device`: Device to use (optional)
- `use_large_calibration`: Use large calibration dataset (optional)

## Tips
- Use descriptive output paths to avoid overwriting files
- Validate your config file before running batch conversions
- Refer to the [examples/batched/README.md](../examples/batched/README.md) for more usage examples 