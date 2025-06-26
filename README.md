# Model Converter Tool

A CLI and API tool for converting, validating, and managing machine learning models across multiple formats. Supports ONNX, FP16, HuggingFace, TorchScript, GGUF, MLX, GPTQ, AWQ, and more.

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

---

## Validation

- The `validate` command supports smart auto-detection of model format (ONNX, GGUF, MLX, TorchScript, HuggingFace, etc.) when using `--model-type auto` (default).
- Validation outputs detailed reasoning about format detection and required files.
- No need to manually specify format for most common cases; the tool will infer and validate accordingly.
- Example:

```bash
python3 model_converter.py validate --local-path ./outputs/my_model_dir
```

---

## Project Structure

```
ModelConverterTool/
├── model_converter_tool/      # Core logic
│   ├── converter.py          # Conversion engine
│   ├── config.py             # Configuration management
│   ├── utils.py              # Utility functions
│   └── validator.py          # Validation logic
├── configs/                  # YAML configuration files
│   ├── model_presets.yaml    # Model presets
│   └── batch_template.yaml   # Batch conversion templates
├── model_converter.py        # CLI entry point
├── requirements.txt          # Dependencies
├── setup.py / pyproject.toml # Packaging
└── README.md
```

---

## Supported Formats

### Input Formats
- Hugging Face models (`hf:model_name`)
- Local model directories
- ONNX models
- TorchScript models
- GGUF models
- MLX models

### Output Formats
- HuggingFace
- ONNX
- TorchScript
- FP16
- GPTQ
- AWQ
- GGUF
- MLX

---

## Installation

```bash
# Clone and install
git clone https://github.com/duoyuncloud/ModelConverterTool.git
cd ModelConverterTool
pip install -e .

# Optional: Install quantization dependencies
pip install auto-gptq awq llama-cpp-python mlx
```

---

## Typical User Workflow

Most users use this tool to convert their own locally trained models (e.g., HuggingFace, PyTorch, ONNX) to other formats for deployment or interoperability. You can convert a single model or batch convert multiple models using a YAML configuration file. Public model pre-download is not required for normal use.

---

## Quick Start

### CLI Usage

```bash
model-converter convert --hf-model gpt2 --output-format onnx --output-path ./outputs/gpt2_onnx
model-converter convert --hf-model gpt2 --output-format fp16 --output-path ./outputs/gpt2_fp16
model-converter convert --hf-model distilbert-base-uncased --output-format gptq --output-path ./outputs/distilbert_gptq --quantization q4_k_m
model-converter batch-convert --config-file configs/batch_template.yaml
model-converter list-formats
model-converter validate --hf-model gpt2 --model-type text-generation
```

### API Usage

```python
from model_converter_tool import ModelConverter

converter = ModelConverter()
success = converter.convert(
    input_source="hf:distilbert-base-uncased",
    output_format="onnx",
    output_path="./outputs/distilbert_onnx",
    model_type="text-classification"
)
print("Conversion success:", success)
```

---

## Batch Conversion & Configuration

You can convert multiple models at once, or use advanced options, by editing a YAML configuration file.

### 1. Copy the template

```bash
cp configs/batch_template.yaml configs/my_batch.yaml
```

### 2. Edit your YAML config

Fill in your own model paths, output formats, and options. Example for batch conversion:

```yaml
models:
  my_model_1:
    input: "/path/to/my_model_1"
    output_format: "onnx"
    output_path: "outputs/my_model_1_onnx"
    model_type: "text-classification"
  my_model_2:
    input: "/path/to/my_model_2"
    output_format: "torchscript"
    output_path: "outputs/my_model_2_ts"
    model_type: "text-generation"
```

You can also use a YAML file for advanced single-model conversion:

```yaml
model_name: "my-custom-model"
model_type: "text-generation"
output_format: "onnx"
device: "cuda"
quantization: "q4_k_m"
config:
  max_length: 512
  use_cache: false
```

You can place your YAML config anywhere, but we recommend keeping it in the `configs/` directory for organization.

### 3. Run batch conversion

```bash
python3 model_converter.py batch-convert --config-file configs/my_batch.yaml
```

---

## Offline Mode

You can run the converter in **offline mode** to ensure no downloads are performed and only local model files are used. This is useful for air-gapped, reproducible, or secure environments.

- To enable offline mode, add the `--offline-mode` flag to your CLI command.
- In offline mode, you must provide a local model path (not a HuggingFace model name).
- If you try to use a HuggingFace model (e.g., `--hf-model gpt2`) in offline mode, the tool will show an error and stop.

### CLI Example

```bash
# Convert a local model in offline mode
model-converter convert --local-path /models/my_model --output-format onnx --offline-mode

# Batch convert with offline mode
model-converter batch-convert --config-file configs/batch_template.yaml --offline-mode
```

### API Example

```python
converter.convert(
    input_source="/models/my_model",
    output_format="onnx",
    output_path="outputs/my_model_onnx",
    offline_mode=True
)
```

---

## Supported Model Types

- Text: text-generation, text-classification, text2text-generation
- Vision: image-classification, image-segmentation, object-detection
- Audio: audio-classification, audio-ctc, speech-seq2seq
- Multimodal: vision-encoder-decoder, question-answering
- Specialized: token-classification, multiple-choice, fill-mask

---

## Troubleshooting & FAQ

### Common Issues

1. ONNX Conversion Fails
   - Try different model types or simpler models
   - Use `--device cpu` for compatibility
   - Check model complexity and dependencies
2. Quantization Dependencies
   - Install required packages: `auto-gptq`, `awq`, `llama-cpp-python`, `mlx`
   - Some quantization requires CUDA support (Linux + NVIDIA GPU)

### Frequently Asked Questions

**Q: Do I need to pre-download public models?**  
A: No, unless you want to test or demo. For your own models, just use `--local-path` and point to your local model directory.

**Q: Where should I put my YAML config?**  
A: Anywhere you like! The `configs/` directory is recommended for organization, but you can specify any path with `--config-file`.

**Q: Can I convert models fully offline?**  
A: Yes! Use `--offline-mode` and provide a local model path. No internet connection is required for local models.