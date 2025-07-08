# ModelConverterTool

A modern, professional CLI & API tool for converting, validating, and managing machine learning models across multiple formats. **API First, CLI Native**. Supports ONNX, FP16, HuggingFace, TorchScript, GGUF, MLX, GPTQ, AWQ, and more.

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
- **Validation:** Built-in model validation and compatibility checking
- **API & CLI:** API-first, CLI-native design

## Installation

### 1. Clone the repository

```sh
git clone https://github.com/duoyuncloud/ModelConverterTool.git
cd ModelConverterTool
```

### 2. Install Python dependencies

```sh
pip install --no-cache-dir torch>=2.0.0 transformers>=4.30.0 tokenizers>=0.13.0 accelerate>=0.20.0
pip install -r requirements.txt
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

## CLI Usage (Recommended)

### Command Overview

```sh
model-converter --help                # Show all commands
model-converter formats               # List supported formats
model-converter detect <model>        # Detect model format
model-converter plan <model> <format> --output <path> [options]   # Preview conversion plan
model-converter execute <model> <format> --output <path> [options] # Execute conversion (recommended)
model-converter convert <model> <format> --output <path> [options] # One-step conversion (plan+execute)
model-converter batch <config.yaml>   # Batch conversion via YAML
model-converter status                # Show workspace status
```

### 1. Format Information

```sh
model-converter formats                   # List all supported formats
model-converter formats --matrix          # Show format conversion matrix
model-converter formats --input huggingface   # Show details for input format
model-converter formats --output gguf         # Show details for output format
```

### 2. Model Detection

```sh
model-converter detect ./my_model
model-converter detect bert-base-uncased
```

### 3. Plan-Execute Workflow (Recommended)

#### Step 1: Plan

```sh
model-converter plan bert-base-uncased onnx --output ./outputs/bert.onnx
model-converter plan TinyLlama/TinyLlama-1.1B-Chat-v1.0 gguf --output ./outputs/tinyllama.gguf --quantization q4_k_m
```

#### Step 2: Execute

```sh
model-converter execute bert-base-uncased onnx --output ./outputs/bert.onnx
model-converter execute TinyLlama/TinyLlama-1.1B-Chat-v1.0 gguf --output ./outputs/tinyllama.gguf --quantization q4_k_m
```

### 4. One-Step Conversion (Quick Mode)

```sh
model-converter convert gpt2 mlx --output ./outputs/gpt2.mlx
model-converter convert sshleifer/tiny-gpt2 fp16 --output ./outputs/tiny_gpt2_fp16
model-converter convert facebook/opt-125m gptq --output ./outputs/opt_125m_gptq --quantization 4bit
```

### 5. Advanced Options

```sh
# Specify model type
model-converter plan bert-base-uncased onnx --output ./bert.onnx --type feature-extraction

# Specify device
model-converter execute gpt2 gptq --output ./gpt2-4bit.safetensors --quantization 4bit --device cuda

# Use large calibration set (for quantization)
model-converter plan facebook/opt-125m gptq --output ./opt_125m_gptq --quantization 4bit --use-large-calibration

# Execute with large calibration
model-converter execute facebook/opt-125m gptq --output ./opt_125m_gptq --quantization 4bit --use-large-calibration
```

### 6. 常用转换示例（Basic Conversion & Quantization）

#### Basic Conversion

```sh
# Hugging Face → ONNX
model-converter convert bert-base-uncased onnx --output ./outputs/bert.onnx

# Hugging Face → GGUF (仅支持 Llama/Mistral 类模型)
model-converter convert TinyLlama/TinyLlama-1.1B-Chat-v1.0 gguf --output ./outputs/tinyllama.gguf

# Hugging Face → MLX
model-converter convert gpt2 mlx --output ./outputs/gpt2.mlx

# Hugging Face → FP16
model-converter convert sshleifer/tiny-gpt2 fp16 --output ./outputs/tiny_gpt2_fp16

# Hugging Face → TorchScript
model-converter convert bert-base-uncased torchscript --output ./outputs/bert.pt

# Hugging Face → SafeTensors
model-converter convert gpt2 safetensors --output ./outputs/gpt2_safetensors

# Hugging Face → HF (重新保存)
model-converter convert gpt2 hf --output ./outputs/gpt2_hf
```

#### Quantization

```sh
# GPTQ 量化 (4bit)
model-converter convert facebook/opt-125m gptq --output ./outputs/opt_125m_gptq --quantization 4bit

# GPTQ 量化 (4bit) - 使用大校准集提高质量
model-converter convert facebook/opt-125m gptq --output ./outputs/opt_125m_gptq_high_quality --quantization 4bit --use-large-calibration

# AWQ 量化 (4bit)
model-converter convert facebook/opt-125m awq --output ./outputs/opt_125m_awq --quantization 4bit

# AWQ 量化 (4bit) - 使用大校准集提高质量
model-converter convert facebook/opt-125m awq --output ./outputs/opt_125m_awq_high_quality --quantization 4bit --use-large-calibration

# GGUF 量化
model-converter convert TinyLlama/TinyLlama-1.1B-Chat-v1.0 gguf --output ./outputs/tinyllama-1.1b-chat-v1.0.gguf --quantization q4_k_m

# MLX 量化
model-converter convert gpt2 mlx --output ./outputs/gpt2.mlx --quantization q4_k_m
```

**量化质量说明：**
- `--use-large-calibration`: 使用更大的校准数据集进行量化，可以提高量化质量但会增加转换时间
- 适用于 GPTQ 和 AWQ 量化格式
- 建议在追求高质量量化时使用此选项

### 7. Batch Conversion

#### YAML Configuration Example

Create `configs/batch_example.yaml`:

```yaml
tasks:
  - model_path: "meta-llama/Llama-2-7b-hf"
    output_format: "gguf"
    output_path: "./outputs/llama2-7b.gguf"
    quantization: "q4_k_m"
    model_type: "text-generation"
    device: "cpu"
  
  - model_path: "bert-base-uncased"
    output_format: "onnx"
    output_path: "./outputs/bert.onnx"
    model_type: "feature-extraction"
    device: "cpu"
  
  - model_path: "gpt2"
    output_format: "mlx"
    output_path: "./outputs/gpt2.mlx"
    quantization: "q4_k_m"
    model_type: "text-generation"
    device: "cpu"
  
  - model_path: "facebook/opt-125m"
    output_format: "gptq"
    output_path: "./outputs/opt_125m_gptq_high_quality"
    quantization: "4bit"
    model_type: "text-generation"
    device: "cpu"
    use_large_calibration: true
```

#### Batch Commands

```sh
model-converter batch configs/batch_example.yaml
model-converter batch configs/batch_example.yaml --max-workers 4
model-converter batch configs/batch_example.yaml --validate-only
model-converter batch configs/batch_example.yaml --max-retries 3
```

### 8. Supported Formats

#### Input Formats
- **Hugging Face**: 包含 `config.json` 和模型文件的目录
- **ONNX**: `.onnx` 文件
- **GGUF**: `.gguf` 文件 (llama.cpp 格式)
- **TorchScript**: `.pt`, `.pth` 文件
- **SafeTensors**: `.safetensors` 文件

#### Output Formats
- **ONNX**: 跨平台推理标准
- **GGUF**: llama.cpp 优化格式
- **TorchScript**: PyTorch 优化格式
- **FP16**: 半精度格式
- **GPTQ**: GPTQ 量化格式
- **AWQ**: AWQ 量化格式
- **SafeTensors**: 安全存储格式
- **MLX**: Apple Silicon 优化格式

#### Quantization Options
- **GGUF**: `q4_k_m`, `q8_0`, `q5_k_m`, `q4_0`, `q4_1`
- **GPTQ**: `4bit`, `8bit`
- **AWQ**: `4bit`, `8bit`
- **MLX**: `q4_k_m`, `q8_0`, `q5_k_m`

## API Usage (For Advanced Users)

> **推荐优先使用CLI。API适合集成或高级定制场景。**

```python
from model_converter_tool.api import ModelConverterAPI
api = ModelConverterAPI()

# Detect model format
detect_info = api.detect_model("gpt2")
print(detect_info)

# Plan and execute conversion
plan = api.plan_conversion("gpt2", "onnx", "./outputs/gpt2.onnx", model_type="text-generation", device="cpu")
result = api.execute_conversion(plan)
print(result.success, result.output_path)

# Batch conversion (see configs/batch_example.yaml)
# ...
```

## Why Plan-Execute?
- **安全**：先预览计划，确认无误再执行，防止误操作
- **透明**：提前看到资源消耗、兼容性、潜在风险
- **专业**：行业最佳实践，适合大模型/大数据场景

## Design Philosophy
- **API First**：所有核心逻辑都通过API实现，CLI只是友好包装
- **CLI Native**：CLI体验现代、直观、专业，支持plan-execute安全模式

---

如需更多帮助，请使用 `model-converter --help` 或查阅文档。