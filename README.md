# ModelConverterTool

A powerful, flexible CLI and API tool for converting machine learning models between different formats. Supports a wide range of model types including text generation, classification, vision, and audio models.

## 🚀 Features

- **Multi-Format Support**: Convert between HuggingFace, ONNX, GGUF, MLX, TorchScript, FP16, GPTQ, AWQ, and SafeTensors formats
- **Model Type Coverage**: Text generation, classification, vision, audio, and more
- **Quantization Support**: Built-in support for GPTQ, AWQ, and GGUF quantization
- **Cross-Platform**: Works on CPU and GPU (CUDA/MPS) with automatic device detection
- **Batch Processing**: Convert multiple models at once using YAML configuration
- **Validation**: Built-in model validation and compatibility checking
- **Performance Optimized**: Caching, parallel processing, and memory-efficient loading

## 📦 Installation

### Quick Install (Recommended)

```bash
# Clone the repository
git clone https://github.com/duoyuncloud/ModelConverterTool.git
cd ModelConverterTool

# Install dependencies in the correct order
./install_dependencies.sh

# Install the package
pip install -e .
```

### Manual Installation

```bash
# 1. Install core dependencies first
pip install -r requirements-core.txt

# 2. Install optional dependencies
pip install -r requirements-optional.txt

# 3. Install the package
pip install -e .
```

## 🎯 Quick Start

### Basic Usage

```bash
# Convert a model to ONNX format
model-converter convert gpt2 onnx --output-path ./outputs/gpt2.onnx

# Convert with quantization
model-converter convert meta-llama/Llama-2-7b-hf gguf --output-path ./outputs/llama2-7b.gguf

# Convert to MLX format (for Apple Silicon)
model-converter convert bert-base-uncased mlx --output-path ./outputs/bert.mlx
```

### Supported Formats

| Input Formats | Output Formats |
|---------------|----------------|
| HuggingFace (`hf:`) | HuggingFace (`hf`) |
| Local models | ONNX (`onnx`) |
| ONNX models | GGUF (`gguf`) |
| GGUF models | MLX (`mlx`) |
| MLX models | TorchScript (`torchscript`) |
| TorchScript | FP16 (`fp16`) |
| SafeTensors | GPTQ (`gptq`) |
| | AWQ (`awq`) |
| | SafeTensors (`safetensors`) |

### Model Types

- **Text Generation**: GPT-2, Llama, DialoGPT, etc.
- **Text Classification**: BERT, DistilBERT, etc.
- **Vision**: ViT, ResNet, etc.
- **Audio**: Speech recognition, classification
- **Multimodal**: Vision-language models

## 🔧 Advanced Usage

### Batch Conversion

Create a YAML configuration file:

```yaml
# configs/my_batch.yaml
models:
  gpt2_to_onnx:
    input: "hf:gpt2"
    output_format: "onnx"
    output_path: "outputs/gpt2.onnx"
    model_type: "text-generation"
    device: "cpu"
  
  bert_to_gguf:
    input: "hf:bert-base-uncased"
    output_format: "gguf"
    output_path: "outputs/bert.gguf"
    model_type: "text-classification"
    quantization: "q4_k_m"
```

Run batch conversion:

```python
from model_converter_tool.converter import ModelConverter

converter = ModelConverter()
converter.batch_convert_from_yaml("configs/my_batch.yaml")
```

### API Usage

```python
from model_converter_tool.converter import ModelConverter

# Initialize converter
converter = ModelConverter()

# Convert model
result = converter.convert(
    input_source="gpt2",
    output_format="onnx",
    output_path="./outputs/gpt2.onnx",
    model_type="text-generation",
    device="cpu",
    validate=True
)

print(f"Conversion successful: {result['success']}")
```

### Quantization Options

```bash
# GPTQ quantization (4-bit, 128 group size)
model-converter convert llama2 gguf --output-path ./llama2-q4.gguf

# AWQ quantization
model-converter convert model awq --output-path ./model-awq

# Custom quantization
model-converter convert model gguf --output-path ./model-custom.gguf
```

## 📋 Examples

### Convert GPT-2 to ONNX
```bash
model-converter convert gpt2 onnx --output-path ./gpt2.onnx
```

### Convert BERT to GGUF with Quantization
```bash
model-converter convert bert-base-uncased gguf --output-path ./bert-q4.gguf
```

### Convert Vision Model to MLX
```bash
model-converter convert google/vit-base-patch16-224 mlx --output-path ./vit.mlx
```

### Convert Large Language Model to FP16
```bash
model-converter convert meta-llama/Llama-2-7b-hf fp16 --output-path ./llama2-fp16
```

## 🏗️ Architecture

The tool is built with modularity and extensibility in mind:

- **Core Converter**: Handles format-specific conversion logic
- **Model Loader**: Intelligent model loading with fallback strategies
- **Validator**: Model validation and compatibility checking
- **CLI Interface**: User-friendly command-line interface
- **Configuration System**: YAML-based configuration management