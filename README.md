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

> **MLX 支持（仅限 macOS arm64/Apple Silicon）：**
> 如需使用 MLX 相关功能，请在 Apple Silicon Mac 上手动安装：
> ```bash
> pip install mlx
> ```
> 或（如支持 extra_requires）：
> ```bash
> pip install .[mlx]
> ```

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

Using `facebook/opt-125m` for quick testing:

```bash
# GPTQ quantization (quick test, 默认小校准集)
model-converter convert facebook/opt-125m gptq --output-path ./outputs/opt_125m_gptq

# AWQ quantization (quick test, 默认小校准集)
model-converter convert facebook/opt-125m awq --output-path ./outputs/opt_125m_awq

# GGUF with quantization
model-converter convert facebook/opt-125m gguf --output-path ./outputs/opt_125m_q4.gguf
```

#### High-Precision Quantization (CLI)

For best quantization quality, you can enable automatic large calibration dataset (256 samples from WikiText-103, each >256 tokens) via CLI:

```bash
# GPTQ high-precision
model-converter convert facebook/opt-125m gptq --output-path ./outputs/opt_125m_gptq --use-large-calibration
# AWQ high-precision
model-converter convert facebook/opt-125m awq --output-path ./outputs/opt_125m_awq --use-large-calibration
```

- `--use-large-calibration` will automatically use a large, high-quality calibration dataset for quantization (slower, recommended for production or final model export).

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

# --- Quantization Examples ---
# Quick quantization (default small calibration set, for testing)
result = converter.convert(
    input_source="facebook/opt-125m",
    output_format="gptq",
    output_path="outputs/opt_125m_gptq_quantized",
    model_type="text-generation",
    device="auto",
    validate=True,
    quantization_config={
        "damp_percent": 0.015,
        # No calibration_dataset: uses small built-in set
    }
)

# AWQ quick quantization
result = converter.convert(
    input_source="facebook/opt-125m",
    output_format="awq",
    output_path="outputs/opt_125m_awq_quantized",
    model_type="text-generation",
    device="auto",
    validate=True,
    quantization_config={
        "damp_percent": 0.015,
    }
)

# High-precision quantization (auto large calibration set, for production)
result = converter.convert(
    input_source="facebook/opt-125m",
    output_format="gptq",
    output_path="outputs/opt_125m_gptq_quantized",
    model_type="text-generation",
    device="auto",
    validate=True,
    quantization_config={
        "damp_percent": 0.015,
    },
    use_large_calibration=True,  # Enable large calibration dataset
)

# AWQ high-precision quantization
result = converter.convert(
    input_source="facebook/opt-125m",
    output_format="awq",
    output_path="outputs/opt_125m_awq_quantized",
    model_type="text-generation",
    device="auto",
    validate=True,
    quantization_config={
        "damp_percent": 0.015,
    },
    use_large_calibration=True,
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

## 安装与使用建议

- 推荐使用如下命令安装本工具包：

```sh
pip install .
# 或
pip install model-converter-tool
```

- 安装后可直接在命令行使用 `model-converter` 或在 Python 中 `import model_converter_tool`。

- **不建议**直接运行源码目录下的测试脚本（如 `python tests/test_xxx.py`），否则可能遇到 `ModuleNotFoundError: No module named 'model_converter_tool'`。

- 如遇 import 错误，可先执行 `pip install .`，或在测试脚本开头加：

```python
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
```

## 常见问题（FAQ）

### Q: Windows 下提示 'model-converter' 不是内部或外部命令？
A: 可能是 Python 的 Scripts 目录未加入 PATH。可用如下命令代替：

```sh
python -m model_converter_tool.cli [参数]
```

其它平台如遇类似问题也可用此方法。