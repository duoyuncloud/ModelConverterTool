# Model Converter Tool Documentation

A flexible, extensible tool for converting, quantizing, and managing machine learning models across multiple formats and frameworks.

## Features

- **Multi-format support**: GGUF, ONNX, HuggingFace, Safetensors, TorchScript, MLX, GPTQ, AWQ, and more
- **muP-to-LLaMA scaling**: Automatically detect and rescale muP-initialized models for LLaMA compatibility
- **Fake weights**: Generate models with zero or custom-shaped weights for testing and debugging
- **Advanced quantization**: Fine-grained control with custom quantization configurations
- **Batch processing**: Convert multiple models using YAML/JSON configuration files
- **API-first design**: Use via CLI, Python API, or integrate into your workflows

## Quick Start

### Installation
```bash
# Install system dependencies (macOS/Linux)
./install.sh --system-deps

# Install Python dependencies
./install.sh

# For Apple Silicon users (enable MLX support)
./install.sh --enable-mlx
```

### Basic Usage
```bash
# Convert a single model
modelconvert convert gpt2 onnx -o outputs/gpt2_onnx

# Convert with quantization
modelconvert convert facebook/opt-125m gptq --quant 4bit -o outputs/opt_gptq

# Convert muP model to LLaMA format
modelconvert convert path/to/mup_model safetensors --mup2llama -o outputs/llama_model

# Generate fake weight model for testing
modelconvert convert gpt2 safetensors --fake-weight -o outputs/fake_model

# Batch convert multiple models
modelconvert batch configs/batch_config.yaml

# Inspect model details
modelconvert inspect gpt2

# Check model usability
modelconvert check outputs/model.onnx

# View conversion history
modelconvert history
```

## Documentation

- **[CLI Reference](./cli.md)** - Complete command-line interface documentation
- **[Configuration](./config.md)** - Batch processing and advanced configuration options
- **[Converter Engine](./converter.md)** - Technical details and supported formats

## Examples

Explore example scripts in the `examples/` directory:
- [Basic conversion](../examples/example_basic.py)
- [Batch processing](../examples/example_batch.py)
- [Quantization](../examples/example_quantization.py)
- [API usage](../examples/example_api.py)
- [muP-to-LLaMA](../examples/example_mup2llama.py)
- [Fake weights](../examples/example_fake_weight.py)

## Contributing

Please see [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines.

## License

Licensed under the Apache 2.0 License. 