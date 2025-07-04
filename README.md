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
model-converter convert TinyLlama/TinyLlama-1.1B-Chat-v1.0 gguf --output-path ./outputs/tinyllama-1.1b-chat-v1.0.gguf

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
    input_source="gpt2",
    output_format="onnx",
    output_path="./outputs/gpt2.onnx",
    model_type="text-generation",
    device="cpu",
    validate=True
)
print(result.success, result.output_path)
```

## FAQ

- **Q: Windows 下提示 'model-converter' 不是内部或外部命令？**
  - A: 可能是 Python 的 Scripts 目录未加入 PATH。可用如下命令代替：
    ```sh
    python -m model_converter_tool.cli [参数]
    ```

- **Q: 直接运行源码目录下的测试脚本报错？**
  - A: 推荐先执行 `pip install .`，或在测试脚本开头加：
    ```python
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    ```

## See Also
- See `tests/README.md` for all supported conversion and quantization test cases.

# 示例：将 HuggingFace gpt2 模型转为 MLX

假设你要将 gpt2 转换为 MLX 格式：

```bash
python -m model_converter_tool.cli \
  --model_path gpt2 \
  --output_format mlx \
  --output_path ./outputs/gpt2.mlx
```

> gpt2 模型无需 HuggingFace 权限，适合测试和开发流程。