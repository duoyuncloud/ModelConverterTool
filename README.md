# ModelConverterTool

A CLI and API tool for converting, validating, and managing machine learning models across multiple formats. Supports ONNX, FP16, HuggingFace, TorchScript, GGUF, MLX, GPTQ, AWQ, and more.

## Project Structure

- `model_converter_tool/` — Core library and CLI implementation
- `tests/` — Minimal tests, strictly corresponding to README examples
- `configs/` — Example YAML batch configs
- `outputs/` — Output directory for converted models (created at runtime)

## Features
- **Multi-Format Support:** ONNX, GGUF, MLX, TorchScript, FP16, GPTQ, AWQ, SafeTensors, HuggingFace
- **Quantization:** Built-in GPTQ, AWQ, and GGUF quantization support
- **Batch Processing:** Convert multiple models using YAML configuration
- **Cross-Platform:** CPU and GPU (CUDA/MPS) with automatic device detection
- `Validation:** Built-in model validation and compatibility checking
- **API & CLI:** Both programmatic and command-line interfaces

## Installation

### 1. Clone the repository

```sh
git clone https://github.com/duoyuncloud/ModelConverterTool.git
cd ModelConverterTool
```

### 2. Install Python dependencies

```sh
pip install -e .
```

> **MLX support (macOS arm64/Apple Silicon only):**
> To use MLX features, install MLX manually:
> ```sh
> pip install mlx
> ```
> Or (if supported):
> ```sh
> pip install .[mlx]
> ```

### 3. (Optional) Install system dependencies

For best compatibility, run:

```sh
chmod +x install_system_deps.sh
./install_system_deps.sh
```

This will install system tools like git, make, python3, cmake, etc.

## Usage Examples

### 1. Basic Model Format Conversion (CLI)

```sh
# bert-base-uncased → onnx
model-converter convert bert-base-uncased onnx --output-path ./outputs/bert.onnx

# TinyLlama-1.1B-Chat-v1.0 → gguf
#(SLOW loading) model-converter convert TinyLlama/TinyLlama-1.1B-Chat-v1.0 gguf --output-path ./outputs/tinyllama-1.1b-chat-v1.0.gguf  # Only Llama/Mistral-like models are supported for GGUF. Most Hugging Face models (e.g., GPT2, BERT, OPT) are NOT supported by the official GGUF converter.

# gpt2 → mlx
model-converter convert gpt2 mlx --output-path ./outputs/gpt2.mlx

# tiny-gpt2 → fp16
model-converter convert sshleifer/tiny-gpt2 fp16 --output-path ./outputs/tiny_gpt2_fp16

# bert-base-uncased → torchscript
model-converter convert bert-base-uncased torchscript --output-path ./outputs/bert.pt

# gpt2 → safetensors
model-converter convert gpt2 safetensors --output-path ./outputs/gpt2_safetensors

# gpt2 → hf (HuggingFace format)
model-converter convert gpt2 hf --output-path ./outputs/gpt2_hf
```

### 2. Quantization

```sh
# GPTQ quantization (quick test)
model-converter convert facebook/opt-125m gptq --output-path ./outputs/opt_125m_gptq

# AWQ quantization (quick test)
model-converter convert facebook/opt-125m awq --output-path ./outputs/opt_125m_awq

# GGUF with quantization
model-converter convert facebook/opt-125m gguf --output-path ./outputs/opt_125m_q4.gguf
```

### 3. Batch Conversion (YAML)

Create a YAML config (e.g. `configs/gpt2_batch.yaml`):

```yaml
models:
  bert_to_onnx:
    input: "bert-base-uncased"
    output_format: "onnx"
    output_path: "outputs/bert.onnx"
    model_type: "feature-extraction"
    device: "cpu"
  tiny_gpt2_to_fp16:
    input: "sshleifer/tiny-gpt2"
    output_format: "fp16"
    output_path: "outputs/tiny_gpt2_fp16"
    model_type: "text-generation"
    device: "cpu"
  gpt2_to_torchscript:
    input: "gpt2"
    output_format: "torchscript"
    output_path: "outputs/gpt2.pt"
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

```python
from model_converter_tool.converter import ModelConverter
converter = ModelConverter()

# Basic conversion
result = converter.convert(
    model_name="gpt2",
    output_format="onnx",
    output_path="./outputs/gpt2.onnx",
    model_type="text-generation",
    device="cpu",
    validate=True
)
print(result.success, result.output_path)
```