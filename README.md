# ModelConverterTool

A professional, multi-format machine learning model conversion and management tool. Supports ONNX, GGUF, MLX, TorchScript, FP16, GPTQ, AWQ, SafeTensors, HuggingFace and more. Built-in quantization, validation, and batch processing. **Clean, orthogonal CLI. Easy-to-integrate API.**

---

## Installation

```sh
git clone https://github.com/duoyuncloud/ModelConverterTool.git
cd ModelConverterTool
pip install -e .
# For MLX support (Apple Silicon only):
pip install mlx
```
---
## CLI Quick Start

### Command Overview

```sh
modelconvert inspect <model>                # Inspect model format and info
modelconvert convert <input> [--output OUTPUT] [--to FORMAT] [--quant QUANT] [--device cpu/cuda]  # Convert model
modelconvert list [formats|quantizations]   # List supported formats/quantization types
modelconvert validate <model> [--output-format FORMAT]  # Validate model or conversion feasibility
modelconvert cache                          # Show local cache and task status
modelconvert history                        # Show conversion history
modelconvert config [show|get|set|list_presets] [key] [value]  # Config management
modelconvert version                        # Show version
```

> **Tip:**
> Run `modelconvert --help` to see all available commands and options, or `modelconvert <command> --help` for details on a specific command.
> 
> **Output path auto-completion:**
> If you omit the output path, it will be auto-generated based on the input and target format. For file formats, the correct extension is added automatically; for directory formats, a suitable directory is created.

### Common Examples

```sh
# Inspect model format
modelconvert inspect meta-llama/Llama-2-7b-hf

# ONNX conversion (HuggingFace Hub, output path auto-completed)
modelconvert convert bert-base-uncased --to onnx

# ONNX conversion (local model, MUST specify --output)
modelconvert convert bert-base-uncased --output ./outputs/bert.onnx --to onnx

# Quantized conversion (output path auto-completed)
modelconvert convert facebook/opt-125m --to gptq --quant 4bit

# Quantized conversion with custom output path
modelconvert convert facebook/opt-125m --output ./outputs/opt_125m_gptq --to gptq --quant 4bit

# List supported formats
modelconvert list formats

# List quantization types
modelconvert list quantizations

# Validate model file
modelconvert validate ./outputs/llama-2-7b.gguf

# Show history and cache
modelconvert history
modelconvert cache

# Config management
modelconvert config show
modelconvert config set cache_dir ./mycache
```

> **Warning:**
> - GGUF conversion only supports Llama/Mistral/Gemma family models (e.g., meta-llama/Llama-2-7b-hf, TinyLlama/TinyLlama-1.1B-Chat-v1.0, arnir0/Tiny-LLM). OPT, GPT2, BERT, etc. are **not** supported for GGUF by llama.cpp.
> - **If you use a local model path (e.g., ./models/llama.bin), you MUST specify --output.** Otherwise, the tool will treat it as a HuggingFace repo id and report an error.

---

## Basic Conversion Examples

```sh
# Hugging Face → ONNX
modelconvert convert bert-base-uncased --output ./outputs/bert.onnx --to onnx

# Hugging Face → GGUF (Llama/Mistral family)
modelconvert convert arnir0/Tiny-LLM --output ./outputs/tiny-llm.gguf --to gguf

# Hugging Face → GGUF (Llama/Mistral family)
modelconvert convert arnir0/Tiny-LLM --output ./outputs/tiny-llm.gguf --to gguf

# Hugging Face → MLX
modelconvert convert gpt2 --output ./outputs/gpt2.mlx --to mlx

# Hugging Face → FP16
modelconvert convert sshleifer/tiny-gpt2 --output ./outputs/tiny_gpt2_fp16 --to fp16

# Hugging Face → TorchScript
modelconvert convert bert-base-uncased --output ./outputs/bert.pt --to torchscript

# Hugging Face → SafeTensors
modelconvert convert gpt2 --output ./outputs/gpt2_safetensors --to safetensors

# Hugging Face → HF (re-save)
modelconvert convert gpt2 --output ./outputs/gpt2_hf --to hf
```

---

## Quantization Demo

```sh
# GPTQ quantization (4bit)
modelconvert convert facebook/opt-125m --output ./outputs/opt_125m_gptq --to gptq --quant 4bit

# GPTQ quantization (4bit, high quality)
modelconvert convert facebook/opt-125m --output ./outputs/opt_125m_gptq_high_quality --to gptq --quant 4bit --use-large-calibration

# AWQ quantization (4bit)
modelconvert convert facebook/opt-125m --output ./outputs/opt_125m_awq --to awq --quant 4bit

# AWQ quantization (4bit, high quality)
modelconvert convert facebook/opt-125m --output ./outputs/opt_125m_awq_high_quality --to awq --quant 4bit --use-large-calibration

# GGUF quantization (Llama/Mistral/Gemma only)
modelconvert convert arnir0/Tiny-LLM --output ./outputs/tiny-llm.gguf --to gguf --quant q4_k_m

# MLX quantization
modelconvert convert gpt2 --output ./outputs/gpt2.mlx --to mlx --quant q4_k_m
```

> **Note:**
> GGUF quantization only supports Llama/Mistral/Gemma family models. Attempting GGUF conversion on other architectures will fail.

---

## Supported Formats & Quantization

- **Input formats:** HuggingFace, ONNX, GGUF, TorchScript, SafeTensors
- **Output formats:** ONNX, GGUF, TorchScript, FP16, GPTQ, AWQ, SafeTensors, MLX
- **Quantization options:** GGUF/MLX (`q4_k_m` etc.), GPTQ/AWQ (`4bit`/`8bit`)

---

## Design Philosophy

- **Orthogonal & Simple:** Each command does one thing, clear parameters.
- **Professional & Reliable:** Built-in format detection, conversion, quantization, validation, batch processing.
- **Extensible:** API/CLI layered, easy to integrate and extend.

---

## API Usage (Advanced)

```python
from model_converter_tool.api import ModelConverterAPI
api = ModelConverterAPI()
info = api.detect_model("gpt2")
result = api.converter.convert(model_name="gpt2", output_format="onnx", output_path="./gpt2.onnx")
```

---

## Help & Documentation

See `modelconvert --help` or the `docs/` directory for more usage details.

## Running All Tests

To run all tests, execute:

```bash
./run_all_tests.sh
```