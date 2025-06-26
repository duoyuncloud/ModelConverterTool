# Model Converter Tool

A powerful CLI tool for converting Hugging Face models between different formats with real conversion capabilities and standard HF format compliance.

## Features

- **Real Model Conversions**: Actual conversion logic for all supported formats
- **Standard HF Format**: All outputs comply with Hugging Face format standards
- **Multi-format Support**: Convert between ONNX, TorchScript, FP16, GPTQ, AWQ, GGUF, and MLX
- **Quantization Support**: Built-in quantization for GPTQ, AWQ, and GGUF formats
- **Fallback Strategies**: Multi-step fallback for complex model conversions
- **Batch Operations**: Convert multiple models using YAML configurations
- **Offline Mode**: Work with local models without internet access

## Project Structure

```
Model-Converter-Tool/
├── src/                    # Core conversion logic
│   ├── converter.py       # Main conversion engine
│   ├── config.py          # Configuration management
│   ├── utils.py           # Utility functions
│   └── validator.py       # Validation logic
├── configs/               # YAML configuration files
│   ├── model_presets.yaml # Model presets and configurations
│   └── batch_template.yaml # Batch conversion templates
├── model_converter.py     # CLI entry point
└── requirements-cli.txt   # Dependencies
```

## Supported Formats

### Input Formats
- Hugging Face models (`hf:model_name`)
- Local model directories
- ONNX models
- TorchScript models
- GGUF models
- MLX models

### Output Formats
- **HF**: Standard Hugging Face format with optimizations
- **ONNX**: Optimized for inference with dynamic shapes
- **TorchScript**: PyTorch's production-ready format
- **FP16**: Half-precision floating point for memory efficiency
- **GPTQ**: 4-bit quantization for large language models
- **AWQ**: Activation-aware quantization
- **GGUF**: Optimized for llama.cpp inference
- **MLX**: Apple Silicon optimized format

## Installation

```bash
# Clone and install
git clone <repository-url>
cd Model-Converter-Tool-clean
pip install -r requirements-cli.txt

# Optional: Install quantization dependencies
pip install auto-gptq awq llama-cpp-python mlx
```

## Quick Start

### Basic Conversion

```bash
# Convert to HF format (optimized)
python model_converter.py convert \
    --input "hf:distilbert-base-uncased" \
    --output-format hf \
    --output-path "./outputs/distilbert_hf" \
    --model-type text-classification

# Convert to ONNX
python model_converter.py convert \
    --input "hf:distilbert-base-uncased" \
    --output-format onnx \
    --output-path "./outputs/distilbert_onnx" \
    --model-type text-classification

# Convert to FP16
python model_converter.py convert \
    --input "hf:gpt2" \
    --output-format fp16 \
    --output-path "./outputs/gpt2_fp16" \
    --model-type text-generation

# Convert with quantization
python model_converter.py convert \
    --input "hf:distilbert-base-uncased" \
    --output-format gptq \
    --output-path "./outputs/distilbert_gptq" \
    --model-type text-classification \
    --quantization q4_k_m
```

### Batch Conversion

```bash
# Convert multiple models
python model_converter.py batch \
    --config configs/batch_template.yaml \
    --output-dir "./batch_outputs"
```

### Advanced Options

```bash
# Use specific device
python model_converter.py convert \
    --input "hf:model" \
    --output-format onnx \
    --output-path "./output" \
    --device cuda

# Offline mode
python model_converter.py convert \
    --input "./local_model" \
    --output-format onnx \
    --output-path "./outputs/local_onnx" \
    --offline-mode

# Validate conversion
python model_converter.py validate \
    --input "hf:gpt2" \
    --output-format onnx
```

## Configuration

### Model Presets

The tool includes predefined model configurations in `configs/model_presets.yaml`:

```yaml
common_models:
  bert-base-uncased:
    default_format: onnx
    description: BERT base uncased model
    model_type: text-classification
    supported_formats:
    - onnx
    - gguf
    - mlx
    - torchscript
```

### Custom Configuration

Create custom YAML configurations for specific models:

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

## Output Format

All conversions produce outputs that comply with Hugging Face format standards:

```
output_model/
├── model.onnx          # Converted model file (ONNX)
├── model.pt            # Converted model file (TorchScript)
├── model.safetensors   # Model weights (HF/FP16)
├── config.json         # Model configuration
├── tokenizer.json      # Tokenizer configuration
├── special_tokens_map.json  # Special tokens
├── format_config.json  # Format-specific metadata
└── README.md          # Model card with conversion info
```

### Format-Specific Features

- **HF**: Safe serialization, model type optimization, device-specific optimizations
- **ONNX**: Dynamic shapes, optimized graph, minimal ONNX fallback
- **TorchScript**: Script/trace fallback, use_cache optimization
- **FP16**: Half-precision weights, memory efficient
- **GPTQ/AWQ**: Quantized weights with calibration data
- **GGUF**: llama.cpp optimized format
- **MLX**: Apple Silicon optimized weights

## Supported Model Types

- **Text Models**: text-generation, text-classification, text2text-generation
- **Vision Models**: image-classification, image-segmentation, object-detection
- **Audio Models**: audio-classification, audio-ctc, speech-seq2seq
- **Multimodal**: vision-encoder-decoder, question-answering
- **Specialized**: token-classification, multiple-choice, fill-mask

## Performance Optimizations

### Fast Model Loading
Optimized loading for common models with preset configurations:

```python
fast_models = {
    'gpt2': {'max_length': 512, 'use_cache': False},
    'bert-base-uncased': {'max_length': 512},
    'distilbert-base-uncased': {'max_length': 512},
}
```

### Fallback Strategies
- **ONNX**: transformers.onnx → torch.onnx → minimal ONNX
- **TorchScript**: script → trace → use_cache=False
- **Quantization**: Multiple quantization levels and methods

## Testing

```bash
# Test enhanced conversions
python test_enhanced_conversion.py

# Test local conversions (offline)
python test_local_conversion.py
```

## Troubleshooting

### Common Issues

1. **ONNX Conversion Fails**:
   - Try different model types or simpler models
   - Use `--device cpu` for compatibility
   - Check model complexity and dependencies

2. **Quantization Dependencies**:
   - Install required packages: `auto-gptq`, `awq`, `llama-cpp-python`
   - Some quantization requires CUDA support

3. **Memory Issues**:
   - Use `--device cpu` for large models
   - Try FP16 conversion for memory efficiency
   - Use quantization for very large models

### Debug Mode

```bash
python model_converter.py convert \
    --input "hf:model" \
    --output-format onnx \
    --output-path "./output" \
    --verbose
```