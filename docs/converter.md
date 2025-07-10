# Model Converter Engine

This document provides technical details about the model conversion engine, supported formats, and the internal workflow.

## Overview
The conversion engine is responsible for transforming machine learning models between different formats, applying quantization, and optimizing for various hardware backends.

## Supported Formats
- GGUF
- ONNX
- FP16
- AWQ
- MLX
- GPTQ
- Safetensors
- TorchScript
- ...and more

## Conversion Workflow
1. Load the input model
2. Parse and validate model structure
3. Apply optional quantization or optimization
4. Convert to the target format
5. Save the output model

## Engine Architecture
- Modular design for easy extension
- Pluggable backends for different formats
- Device-aware conversion (CPU, GPU, Apple Silicon)

## Extending the Engine
- How to add support for new model formats
- Guidelines for contributing new conversion modules

## References
- [Core API documentation](../model_converter_tool/api.py)
- [Engine source code](../model_converter_tool/engine/) 