# ModelConverterTool

A professional, API-first and CLI-native tool for machine learning model conversion and management.  
Supports ONNX, GGUF, MLX, TorchScript, FP16, GPTQ, AWQ, SafeTensors, HuggingFace, and more.  
Clean, orthogonal CLI. Easy-to-integrate, extensible API. 

---

## Installation

```sh
git clone https://github.com/duoyuncloud/ModelConverterTool.git
cd ModelConverterTool
./install.sh
source venv/bin/activate
```
Or, for advanced users:
```sh
pip install -e .
pip install -r requirements.txt
```

---

## CLI Commands

| Command | Function |
|---------|----------|
| `modelconvert inspect <model>` | Inspect and display model format and metadata. |
| `modelconvert convert <input> <output> [options]` | Convert a model to another format, with optional quantization. |
| `modelconvert batch <config.yaml> [options]` | Batch convert multiple models using a YAML/JSON config file. |
| `modelconvert history` | Show conversion history (completed, failed, active tasks). |
| `modelconvert config [--action ...] [--key ...] [--value ...]` | Manage tool configuration (show, get, set, list presets). |

---

## Command Details

- **inspect**:  
  Inspect a model file or repo and display its format, metadata, and convertible targets.

- **convert**:  
  Convert a model to another format.  
  - `<input>`: Input model path or repo id (required)
  - `<output>`: Output format (required, e.g. onnx, gguf, mlx, gptq, etc.)
  - `--path`: Output file path (optional, auto-completed if omitted)
  - `--quant`: Quantization type (optional)
  - `--model-type`: Model type (optional)
  - `--device`: Device (cpu/cuda, optional)
  - `--use-large-calibration`: Use large calibration dataset (optional)

- **batch**:  
  Batch convert models using a config file.  
  - `<config.yaml>`: Path to YAML/JSON config file describing conversion tasks
  - `--max-workers`: Number of concurrent workers (default: 1)
  - `--max-retries`: Max retries per task (default: 1)
  - `--skip-disk-check`: Skip disk space check (not recommended)

- **history**:  
  Show all completed, failed, and active conversion tasks.

- **config**:  
  Manage tool configuration.  
  - `--action`: show/get/set/list_presets (default: show)
  - `--key`: Config key (for get/set)
  - `--value`: Config value (for set)

---

## Examples

```sh
# Inspect a model
modelconvert inspect meta-llama/Llama-2-7b-hf

# Convert to ONNX (output path auto-completed)
modelconvert convert bert-base-uncased onnx

# Convert to GGUF with quantization
modelconvert convert Qwen/Qwen2-0.5B gguf --quant q4_k_m --model-type qwen

# Batch conversion
modelconvert batch configs/batch_template.yaml --max-workers 2

# Show conversion history
modelconvert history

# Config management
modelconvert config --action show
modelconvert config --action set --key cache_dir --value ./mycache
```

---

## Basic Conversion Demo

```sh
# Hugging Face → ONNX
modelconvert convert bert-base-uncased onnx --path ./outputs/bert.onnx

# Hugging Face → GGUF (Llama/Mistral family, recommended: Qwen/Qwen2-0.5B)
modelconvert convert Qwen/Qwen2-0.5B gguf --path ./outputs/qwen2-0.5B.gguf --model-type qwen

# Hugging Face → MLX
modelconvert convert gpt2 mlx --path ./outputs/gpt2.mlx

# Hugging Face → FP16
modelconvert convert sshleifer/tiny-gpt2 fp16 --path ./outputs/tiny_gpt2_fp16

# Hugging Face → TorchScript
modelconvert convert bert-base-uncased torchscript --path ./outputs/bert.pt

# Hugging Face → SafeTensors
modelconvert convert gpt2 safetensors --path ./outputs/gpt2_safetensors

# Hugging Face → HF (re-save)
modelconvert convert gpt2 hf --path ./outputs/gpt2_hf
```

---

## Quantization Demo

```sh
# GPTQ quantization (4bit)
modelconvert convert facebook/opt-125m gptq --quant 4bit --path ./outputs/opt_125m_gptq

# GPTQ quantization (4bit, high quality)
modelconvert convert facebook/opt-125m gptq --quant 4bit --use-large-calibration --path ./outputs/opt_125m_gptq_high_quality

# AWQ quantization (4bit)
modelconvert convert facebook/opt-125m awq --quant 4bit --path ./outputs/opt_125m_awq

# AWQ quantization (4bit, high quality)
modelconvert convert facebook/opt-125m awq --quant 4bit --use-large-calibration --path ./outputs/opt_125m_awq_high_quality

# GGUF quantization (Llama/Mistral/Gemma only)
modelconvert convert Qwen/Qwen2-0.5B gguf --quant q4_k_m --path ./outputs/qwen2-0.5B.gguf

# MLX quantization
modelconvert convert gpt2 mlx --quant q4_k_m --path ./outputs/gpt2.mlx
```

> **Note:**
> GGUF quantization only supports Llama/Mistral/Gemma family models. Attempting GGUF conversion on other architectures will fail.

---

## Supported Formats & Quantization

- **Input formats:** HuggingFace, ONNX, GGUF, TorchScript, SafeTensors
- **Output formats:** ONNX, GGUF, TorchScript, FP16, GPTQ, AWQ, SafeTensors, MLX, HF
- **Quantization options:**  
  - GPTQ: 4bit, 8bit  
  - AWQ: 4bit, 8bit  
  - GGUF: q4_k_m, q4_k_s, q5_k_m, q5_k_s, q6_k, q8_0  
  - MLX: q4_k_m, q8_0, q5_k_m

---

## API Example

```python
from model_converter_tool.api import ModelConverterAPI
api = ModelConverterAPI()
info = api.detect_model("gpt2")
result = api.converter.convert(model_name="gpt2", output_format="onnx", output_path="./gpt2.onnx")
```

---

## Documentation & Help

- Run `modelconvert --help` or `modelconvert <command> --help` for details.
- See the `docs/` directory for more.

---

## Testing

```sh
./run_all_tests.sh
```