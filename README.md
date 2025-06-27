# Model Converter Tool

A CLI and API tool for converting, validating, and managing machine learning models across multiple formats.  
**Supports ONNX, FP16, HuggingFace, TorchScript, GGUF, MLX, GPTQ, AWQ, and more.**

---

## Features

- Real model conversion for all supported formats
- API and CLI usage
- Multi-format support: ONNX, TorchScript, FP16, GPTQ, AWQ, GGUF, MLX, HuggingFace
- Quantization for GPTQ, AWQ, GGUF (where supported)
- Batch conversion via YAML configuration
- Offline mode for local models
- Fallback strategies for robust conversion
- Model file and configuration validation
- Automatic correctness validation after conversion

---

## CI and Testing

This project uses **GitHub Actions** for continuous integration:

- **Every push and pull request** triggers CI.
- CI will:
  - Install all dependencies (including quantization: `auto-gptq`, `llama-cpp-python`)
  - Run a full GPT-2 conversion and validation pipeline covering:
    - ONNX, TorchScript, FP16, HuggingFace, GPTQ, GGUF
    - Each format is tested for: conversion success, file existence, model loadability, inference, and (for quantized) quality score
- **Test script**:  
  `tests/test_gpt2_conversion_ci.py`  
  (Can be run locally: `pytest tests/test_gpt2_conversion_ci.py -v`)

**Note:**  
- On CI (Linux, no CUDA), quantization tests will gracefully skip or downgrade if dependencies/hardware are missing.
- On local machines with CUDA/NVIDIA, full quantization validation is performed.

---

## Supported Formats

**Input:**  
- Hugging Face models (`hf:model_name`)
- Local model directories
- ONNX, TorchScript, GGUF, MLX

**Output:**  
- HuggingFace, ONNX, TorchScript, FP16, GPTQ, AWQ, GGUF, MLX

---

## Installation

```bash
git clone https://github.com/duoyuncloud/ModelConverterTool.git
cd ModelConverterTool
pip install -e .
# For quantization and GGUF support:
pip install auto-gptq awq llama-cpp-python mlx
```

---

## Quick Start

**CLI Example:**
```bash
model-converter convert --hf-model gpt2 --output-format onnx --output-path ./outputs/gpt2_onnx
model-converter convert --hf-model gpt2 --output-format gptq --output-path ./outputs/gpt2_gptq --quantization q4_k_m
```

**API Example:**
```python
from model_converter_tool import ModelConverter
converter = ModelConverter()
result = converter.convert(
    input_source="hf:gpt2",
    output_format="onnx",
    output_path="./outputs/gpt2_onnx",
    model_type="text-generation",
    validate=True
)
print(result)
```

---

## Batch Conversion

See `configs/batch_template.yaml` for batch conversion examples.

---

## Validation

- All conversions are automatically validated for correctness (load, inference, output shape, quantization quality)
- Validation is performed both in CLI/API and CI

---

## Version Compatibility

- Python: 3.8–3.11
- torch: 2.0+
- transformers: 4.30+
- onnx: 1.13+
- onnxruntime: 1.14+

---

## Troubleshooting

- For quantization, ensure you have the required dependencies and (for GPTQ) a CUDA-capable GPU
- If a format conversion fails, check the CI logs or run locally with `validate=True` for detailed error messages

---

## More

For advanced usage, YAML config, offline mode, and more, see the full documentation and configs in the repo.