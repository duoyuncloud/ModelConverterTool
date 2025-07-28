# Megatron-LM Integration

Comprehensive guide for converting models between HuggingFace and Megatron-LM formats.

## Overview

The Model Converter Tool provides full bidirectional support for converting models between HuggingFace and Megatron-LM formats. This integration enables seamless interoperability between the two popular model formats, supporting various model architectures including MiniCPM, Llama, and Mistral.

## Supported Models

### MiniCPM
- **Full bidirectional support**: HF ↔ Megatron-LM
- **Architecture**: Transformer with RMSNorm, SwiGLU MLP, Rotary Position Embeddings
- **Optimized weight mapping**: Handles attention and MLP layer differences
- **Configuration preservation**: Maintains model metadata and settings

### Llama/Llama2
- **Full bidirectional support**: HF ↔ Megatron-LM
- **Architecture**: LLaMA-style transformer with grouped-query attention
- **Complete layer mapping**: Attention, MLP, and normalization layers
- **Multi-head attention**: Proper handling of query/key/value projections

### Mistral
- **Full bidirectional support**: HF ↔ Megatron-LM
- **Architecture**: Mistral transformer with sliding window attention
- **Variant support**: Handles different Mistral model variants
- **Optimized conversions**: Efficient weight reordering and mapping

## Conversion Directions

### HuggingFace → Megatron-LM (hf2megatron)

Converts HuggingFace models to Megatron-LM format for distributed training and inference.

```bash
# MiniCPM models
modelconvert convert OpenBMB/MiniCPM4-0.5B hf2megatron --model-type minicpm

# Llama models
modelconvert convert meta-llama/Llama-2-7b hf2megatron --model-type llama

# Mistral models
modelconvert convert mistralai/Mistral-7B-v0.1 hf2megatron --model-type llama
```

**Output Format:**
- `model.pt`: PyTorch state dict containing model weights
- `metadata.json`: Model configuration and metadata

### Megatron-LM → HuggingFace (megatron2hf)

Converts Megatron-LM models back to HuggingFace format for standard inference and deployment.

```bash
# MiniCPM models
modelconvert convert models/megatron_model hf --model-type minicpm

# Llama models  
modelconvert convert models/megatron_model hf --model-type llama

# Mistral models
modelconvert convert models/megatron_model hf --model-type llama
```

**Output Format:**
- `config.json`: HuggingFace model configuration
- `model.safetensors`: Model weights in SafeTensors format
- `generation_config.json`: Generation configuration
- `modeling_*.py`: Model implementation files

## Format Detection

The converter automatically detects input formats:

### HuggingFace Format
- Detected by presence of `config.json` and model weight files
- Supports local directories and HuggingFace Hub models
- Handles various HuggingFace model types

### Megatron-LM Format
- Detected by presence of `model.pt` + `metadata.json`
- Supports custom Megatron-LM checkpoint directories
- Handles both single-file and distributed checkpoints

## Model Architecture Mapping

### Attention Layer Mapping

**HuggingFace → Megatron-LM:**
```
q_proj, k_proj, v_proj → linear_qkv (concatenated)
o_proj → linear_proj
```

**Megatron-LM → HuggingFace:**
```
linear_qkv → q_proj, k_proj, v_proj (split)
linear_proj → o_proj
```

### MLP Layer Mapping

**HuggingFace → Megatron-LM:**
```
gate_proj, up_proj → linear_fc1 (concatenated for SwiGLU)
down_proj → linear_fc2
```

**Megatron-LM → HuggingFace:**
```
linear_fc1 → gate_proj, up_proj (split for SwiGLU)
linear_fc2 → down_proj
```

### Normalization Layers

- **RMSNorm**: Preserved across conversions
- **LayerNorm**: Mapped appropriately for each architecture
- **Position embeddings**: Rotary embeddings handled correctly

## Configuration Parameters

### Required Parameters

- `--model-type`: Specifies the model architecture
  - `minicpm`: For MiniCPM models
  - `llama`: For Llama/Llama2/Mistral models

### Optional Parameters

- `--output-path`: Custom output directory
- `--dtype`: Output precision (fp16, fp32)
- `--tokenizer-model`: Path to tokenizer model (if needed)

## Advanced Usage

### Batch Conversion

Convert multiple models using configuration files:

```yaml
models:
  - model_path: OpenBMB/MiniCPM4-0.5B
    output_path: outputs/minicpm_megatron
    output_format: hf2megatron
    model_type: minicpm
  
  - model_path: meta-llama/Llama-2-7b
    output_path: outputs/llama_megatron
    output_format: hf2megatron
    model_type: llama
```

### API Usage

```python
from model_converter_tool.api import ModelConverterAPI

api = ModelConverterAPI()

# HF to Megatron-LM
result = api.convert_model(
    model_name="OpenBMB/MiniCPM4-0.5B",
    output_format="hf2megatron",
    output_path="./minicpm_megatron",
    model_type="minicpm"
)

# Megatron-LM to HF
result = api.convert_model(
    model_name="./megatron_model",
    output_format="hf",
    output_path="./minicpm_hf",
    model_type="minicpm"
)
```

## Technical Details

### Weight Reordering

The converter handles weight reordering for optimal performance:

1. **Attention weights**: Concatenates Q/K/V projections for Megatron-LM
2. **MLP weights**: Concatenates gate/up projections for SwiGLU activation
3. **Bias handling**: Preserves or removes biases based on model configuration
4. **Tensor shapes**: Ensures proper tensor dimensions for each format

### Memory Management

- **Lazy loading**: Models loaded on-demand to minimize memory usage
- **CPU conversion**: All conversions performed on CPU for compatibility
- **Cleanup**: Automatic cleanup of intermediate tensors

### Error Handling

- **Format validation**: Pre-conversion format detection and validation
- **Architecture compatibility**: Checks for supported model types
- **Detailed logging**: Comprehensive error messages and debugging info
- **Graceful failures**: Proper error handling and recovery

## Troubleshooting

### Common Issues

**"Model type not supported"**
- Ensure `--model-type` is set correctly (minicpm, llama)
- Check that the model architecture is supported

**"Format detection failed"**
- Verify input directory contains required files
- For Megatron-LM: Check for `model.pt` and `metadata.json`
- For HuggingFace: Check for `config.json` and weight files

**"Memory error"**
- Use CPU-only conversion (default)
- Ensure sufficient disk space for output
- Consider using smaller model variants for testing

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
# Set debug environment variable
export MODELCONVERT_DEBUG=1

# Run conversion with debug output
modelconvert convert model hf2megatron --model-type minicpm
```

## Performance Considerations

### Conversion Speed
- **CPU-based**: All conversions run on CPU for maximum compatibility
- **Optimized loading**: Efficient model loading and weight mapping
- **Parallel processing**: Batch conversions support multiple workers

### Output Size
- **Megatron-LM**: Compact format with metadata
- **HuggingFace**: Standard format with full configuration
- **Compression**: No additional compression applied

## Integration with Megatron-LM Training

### Training Workflow

1. **Convert to Megatron-LM**: Use `hf2megatron` for training preparation
2. **Train with Megatron-LM**: Use standard Megatron-LM training pipeline
3. **Convert back to HF**: Use `megatron2hf` for deployment

### Distributed Training

- **Tensor parallelism**: Compatible with Megatron-LM tensor parallel training
- **Pipeline parallelism**: Supports pipeline parallel model conversion
- **Checkpointing**: Handles distributed checkpoint formats

## References

- [Megatron-LM Documentation](https://github.com/NVIDIA/Megatron-LM)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Model Converter Tool API](../model_converter_tool/api.py)
- [Megatron Converters](../megatron_converters/) 