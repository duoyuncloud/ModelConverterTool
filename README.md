# ModelConverterTool

A powerful, flexible tool for converting machine learning models between different formats. Supports text generation, classification, vision, and audio models with comprehensive format coverage.

## 🚀 Features

- **Multi-Format Support**: ONNX, GGUF, MLX, TorchScript, FP16, GPTQ, AWQ, SafeTensors, HuggingFace
- **Quantization**: Built-in GPTQ, AWQ, and GGUF quantization support
- **Batch Processing**: Convert multiple models using YAML configuration
- **Cross-Platform**: CPU and GPU (CUDA/MPS) with automatic device detection
- **Validation**: Built-in model validation and compatibility checking
- **API & CLI**: Both programmatic and command-line interfaces

## 📦 Installation

```bash
# Clone and install
git clone https://github.com/duoyuncloud/ModelConverterTool.git
cd ModelConverterTool
pip install -e .
```

## 🎯 Quick Start

### 1. Basic Model Format Conversion

```bash
# bert-base-uncased → onnx
model-converter convert bert-base-uncased onnx --output-path ./outputs/bert.onnx

# gpt2 → gguf
model-converter convert gpt2 gguf --output-path ./outputs/gpt2.gguf

# gpt2 → mlx
model-converter convert gpt2 mlx --output-path ./outputs/gpt2.mlx

# tiny-gpt2 → fp16
model-converter convert sshleifer/tiny-gpt2 fp16 --output-path ./outputs/tiny_gpt2_fp16

# gpt2 → torchscript
model-converter convert gpt2 torchscript --output-path ./outputs/gpt2.pt

# gpt2 → safetensors
model-converter convert gpt2 safetensors --output-path ./outputs/gpt2_safetensors

# gpt2 → hf (HuggingFace format)
model-converter convert gpt2 hf --output-path ./outputs/gpt2_hf
```

### 2. Quantization

Using `sshleifer/tiny-gpt2` for faster testing:

```bash
# GPTQ quantization
model-converter convert sshleifer/tiny-gpt2 gptq --output-path ./outputs/tiny_gpt2_gptq

# AWQ quantization
model-converter convert sshleifer/tiny-gpt2 gptq --output-path ./outputs/tiny_gpt2_gptq

# GGUF with quantization
model-converter convert sshleifer/tiny-gpt2 gguf --output-path ./outputs/tiny_gpt2_q4.gguf
```

### 3. Batch Conversion

Create a YAML configuration file:

```yaml
# configs/gpt2_batch.yaml
models:
  gpt2_to_onnx:
    input: "gpt2"
    output_format: "onnx"
    output_path: "outputs/gpt2.onnx"
    model_type: "text-generation"
    device: "cpu"
  
  gpt2_to_gguf:
    input: "gpt2"
    output_format: "gguf"
    output_path: "outputs/gpt2.gguf"
    model_type: "text-generation"
    device: "cpu"
  
  gpt2_to_fp16:
    input: "gpt2"
    output_format: "fp16"
    output_path: "outputs/gpt2_fp16"
    model_type: "text-generation"
    device: "cpu"
  
  gpt2_to_torchscript:
    input: "gpt2"
    output_format: "torchscript"
    output_path: "outputs/gpt2.pt"
    model_type: "text-generation"
    device: "cpu"
  
  gpt2_to_hf:
    input: "gpt2"
    output_format: "hf"
    output_path: "outputs/gpt2_hf"
    model_type: "text-generation"
    device: "cpu"
```

Run batch conversion:

```python
from model_converter_tool.converter import ModelConverter

converter = ModelConverter()
converter.batch_convert_from_yaml("configs/gpt2_batch.yaml")
```

### 4. API Usage

#### ModelConverter().convert

```python
from model_converter_tool.converter import ModelConverter

converter = ModelConverter()

# Basic conversion
result = converter.convert(
    input_source="gpt2",
    output_format="onnx",
    output_path="./outputs/gpt2.onnx",
    model_type="text-generation",
    device="cpu",
    validate=True
)

# All formats
formats = ["onnx", "gguf", "fp16", "torchscript", "hf"]
for fmt in formats:
    result = converter.convert(
        input_source="gpt2",
        output_format=fmt,
        output_path=f"./outputs/gpt2.{fmt}",
        model_type="text-generation",
        device="cpu",
        validate=True
    )
```

#### ModelConverter().batch_convert

```python
# Batch conversion with gpt2
tasks = [
    {
        "input_source": "gpt2",
        "output_format": "onnx",
        "output_path": "./outputs/batch_gpt2.onnx",
        "model_type": "text-generation",
        "device": "cpu"
    },
    {
        "input_source": "gpt2",
        "output_format": "gguf",
        "output_path": "./outputs/batch_gpt2.gguf",
        "model_type": "text-generation",
        "device": "cpu"
    },
    {
        "input_source": "gpt2",
        "output_format": "fp16",
        "output_path": "./outputs/batch_gpt2_fp16",
        "model_type": "text-generation",
        "device": "cpu"
    },
    {
        "input_source": "gpt2",
        "output_format": "torchscript",
        "output_path": "./outputs/batch_gpt2.pt",
        "model_type": "text-generation",
        "device": "cpu"
    },
    {
        "input_source": "gpt2",
        "output_format": "hf",
        "output_path": "./outputs/batch_gpt2_hf",
        "model_type": "text-generation",
        "device": "cpu"
    }
]

results = converter.batch_convert(tasks, max_workers=2)
```