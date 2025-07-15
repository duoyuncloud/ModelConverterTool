# ModelConverterTool

A professional, **API-first and CLI-native** tool for machine learning model conversion and management.
Supports ONNX, GGUF, MLX, TorchScript, GPTQ, AWQ, SafeTensors (fp16/fp32), HuggingFace, and more.

---

## Installation

```sh
git clone https://github.com/duoyuncloud/ModelConverterTool.git
cd ModelConverterTool
./install.sh
source venv/bin/activate
```

---

## CLI Commands

| Command | Function |
|---------|----------|
| `modelconvert inspect <model>` | Inspect and display model format and metadata. |
| `modelconvert convert <input_model> <output_format> [options]` | Convert a model to another format, with optional quantization. |
| `modelconvert batch <config.yaml> [options]` | Batch convert multiple models using a YAML/JSON config file. |
| `modelconvert history` | Show conversion history. |
| `modelconvert config [--action ...] [--key ...] [--value ...]` | Manage tool configuration. |

---

## Command Details

- **inspect**: Inspect a model file or repo and display its format, metadata, and convertible targets.
- **convert**: Convert a model to another format.
  - `<input_model>`: Input model path or repo id (required)
  - `<output_format>`: Output format (required, e.g. onnx, gguf, mlx, gptq, etc.)
  - `-o`, `--output-path`: Output file path (optional, auto-completed if omitted)
  - `--quant`: Quantization type (optional)
  - `--quant-config`: Advanced quantization config (optional, JSON string or YAML file)
  - `--model-type`: Model type (optional, default: auto)
  - `--device`: Device (optional, default: auto)
  - `--use-large-calibration`: Use large calibration dataset (optional)
  - `--dtype`: Precision for output weights (e.g., fp16, fp32; only for safetensors)
- **batch**: Batch convert models using a config file.
  - `<config.yaml>`: Path to YAML/JSON config file describing conversion tasks
  - `--max-workers`: Number of concurrent workers (default: 1)
  - `--max-retries`: Max retries per task (default: 1)
  - `--skip-disk-check`: Skip disk space check (not recommended)
- **history**: Show all completed, failed, and active conversion tasks.
- **config**: Manage tool configuration.
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

## Fine-grained Quantization Config (GPTQ/AWQ)

You can now use advanced quantization configuration for GPTQ and AWQ engines, supporting options like `bits`, `group_size`, `sym`, `desc_act`, and more. This allows precise control over quantization behavior.

- `desc_act` (bool): Enable or disable the descriptive quantization mechanism for improved accuracy in some models. Default is `True`.

**Example:**

```bash
python -m model_converter_tool.cli convert path/to/input_model gptq -o path/to/output_model-gptq --quant-config '{"bits":4, "group_size":128, "sym":true, "desc_act":true}'
```

Or using a YAML file:

```yaml
# quant.yaml
bits: 4
group_size: 128
sym: true
desc_act: true
```

All parameters in the quantization config will be passed to the quantizer for fine-grained control (unsupported keys will be ignored).

---

## Supported Formats & Quantization

|               | HuggingFace | SafeTensors | TorchScript | ONNX | GGUF | MLX |
|---------------|:-----------:|:-----------:|:-----------:|:----:|:----:|:---:|
| HuggingFace   |      ✓      |      ✓      |      ✓      |  ✓   |  ✓   |  ✓  |
| SafeTensors   |      ✓      |      ✓      |             |      |      |     |
| TorchScript   |             |             |      ✓      |      |      |     |
| ONNX          |             |             |             |  ✓   |      |     |
| GGUF          |             |             |             |      |  ✓   |     |
| MLX           |             |             |             |      |      |  ✓  |

**Quantization options:**
- GPTQ: 4bit, 8bit
- AWQ: 4bit, 8bit
- GGUF: q4_k_m, q5_k_m, q8_0
- MLX: q4_k_m, q8_0, q5_k_m
- SafeTensors: fp16, fp32

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