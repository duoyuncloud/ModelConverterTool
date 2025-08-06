# Tensor Parallel and Pipeline Parallel Converters

This document describes the tensor parallel and pipeline parallel conversion capabilities added to the ModelConverterTool.

## Overview

The tensor parallel and pipeline parallel converters support converting large language models between Megatron and HuggingFace formats when the models are distributed across multiple GPUs using tensor parallelism (TP) and/or pipeline parallelism (PP).

## Supported Models

- **MiniCPM Series**: 0.5B, 1.5B, 3B, 8B, 14B
- **Llama Series**: 7B, 13B, 30B, 65B
- **MiniCPM-4**: With MoE (Mixture of Experts) support

## Key Features

- **Automatic Detection**: Automatically detects model type, size, and parallel configuration
- **Smart Conversion**: Chooses the best conversion strategy based on model characteristics
- **Flexible Configuration**: Supports custom TP/PP configurations
- **Bidirectional Conversion**: Megatron ↔ HuggingFace in both directions
- **Error Handling**: Comprehensive validation and error reporting

## Quick Start

### Basic Usage

```python
from megatron_converters import smart_convert_megatron_to_hf

# One-click conversion with auto-detection
smart_convert_megatron_to_hf(
    checkpoint_path="/path/to/megatron/checkpoint",
    output_path="/path/to/output/hf_weights.pt"
)
```

### Model-Specific Conversion

```python
from megatron_converters import convert_minicpm_8b, convert_llama_7b

# MiniCPM 8B conversion
convert_minicpm_8b(
    checkpoint_path="/path/to/minicpm_8b_checkpoint",
    output_path="/path/to/output/minicpm_8b_hf.pt"
)

# Llama 7B conversion
convert_llama_7b(
    checkpoint_path="/path/to/llama_7b_checkpoint",
    output_path="/path/to/output/llama_7b_hf.pt"
)
```

## Advanced Usage

### Using the TensorParallelConverter Class

```python
from megatron_converters import TensorParallelConverter

converter = TensorParallelConverter()

# Custom configuration
converter.convert_minicpm_megatron_to_hf_tp_pp(
    num_layer=32,
    tp_size=2,
    pp_size=1,
    in_dir="/path/to/megatron/checkpoint",
    save_path="/path/to/output/hf_weights.pt",
    num_kv_heads=8,
    num_query_heads=32
)
```

### Using the SmartConverter Class

```python
from megatron_converters import SmartConverter

converter = SmartConverter()

# Auto-detect model type and size
converter.convert_megatron_to_hf(
    checkpoint_path="/path/to/checkpoint",
    output_path="/path/to/output.pt"
)

# Specify model type and size
converter.convert_megatron_to_hf(
    checkpoint_path="/path/to/checkpoint",
    output_path="/path/to/output.pt",
    model_type="minicpm",
    model_size="8b"
)
```

## Command Line Usage

### Basic TP/PP Conversion

```bash
python -m megatron_converters.tp_pp_converter \
    --model_type minicpm \
    --model_size 8b \
    --in_dir /path/to/megatron/checkpoint \
    --save_path /path/to/output/hf_weights.pt
```

### Custom Configuration

```bash
python -m megatron_converters.tp_pp_converter \
    --model_type minicpm \
    --in_dir /path/to/megatron/checkpoint \
    --save_path /path/to/output/hf_weights.pt \
    --num_layer 32 \
    --tp_size 2 \
    --pp_size 1 \
    --num_kv_heads 8 \
    --num_query_heads 32
```

### Direct Checkpoint Conversion

```bash
python megatron_converters/ckpt_to_hf_minicpm_with_tp_pp.py \
    --num_layer 80 \
    --tp_size 8 \
    --pp_size 4 \
    --in_dir /path/to/megatron/checkpoint \
    --save_path /path/to/output/hf_weights.pt \
    --num_kv_heads 8 \
    --num_query_heads 64
```

## Model Configurations

### MiniCPM Series

| Model Size | Layers | Default TP | Default PP | KV Heads | Query Heads |
|------------|--------|------------|------------|----------|-------------|
| 0.5B       | 12     | 1          | 1          | 8        | 32          |
| 1.5B       | 18     | 1          | 1          | 8        | 32          |
| 3B         | 24     | 1          | 1          | 8        | 32          |
| 8B         | 32     | 2          | 1          | 8        | 32          |
| 14B        | 40     | 4          | 1          | 8        | 32          |

### Llama Series

| Model Size | Layers | Default TP | Default PP | KV Heads | Query Heads |
|------------|--------|------------|------------|----------|-------------|
| 7B         | 32     | 1          | 1          | 8        | 32          |
| 13B        | 40     | 2          | 1          | 8        | 40          |
| 30B        | 60     | 4          | 1          | 8        | 52          |
| 65B        | 80     | 8          | 1          | 8        | 64          |

## Parallel Configuration Validation

The converters automatically validate parallel configurations:

- **Query heads** must be divisible by **KV heads**
- **Number of layers** must be divisible by **PP size**
- **Query heads** must be divisible by **TP size**
- **KV heads** must be divisible by **TP size**

## Error Handling

The converters provide comprehensive error handling:

```python
try:
    smart_convert_megatron_to_hf(input_path, output_path)
except FileNotFoundError as e:
    print(f"Checkpoint not found: {e}")
except AssertionError as e:
    print(f"Invalid configuration: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## File Structure

### Input Megatron Checkpoint Structure

```
checkpoint_dir/
├── mp_rank_00/
│   └── model_optim_rng.pt
├── mp_rank_01/
│   └── model_optim_rng.pt
└── ...
```

For pipeline parallelism:
```
checkpoint_dir/
├── mp_rank_00_000/
│   └── model_optim_rng.pt
├── mp_rank_00_001/
│   └── model_optim_rng.pt
├── mp_rank_01_000/
│   └── model_optim_rng.pt
└── ...
```

### Output HuggingFace Structure

```
output.pt  # Single file containing all weights
```

## Performance Considerations

- **Memory Usage**: Large models may require significant memory for conversion
- **Disk Space**: Ensure sufficient disk space for both input and output
- **Processing Time**: Conversion time scales with model size and parallel configuration

## Troubleshooting

### Common Issues

1. **File Not Found**: Ensure checkpoint directory exists and contains expected files
2. **Invalid Configuration**: Check that TP/PP sizes are compatible with model parameters
3. **Memory Errors**: Use smaller batch sizes or process on machines with more memory
4. **Import Errors**: Ensure all dependencies are installed

### Debug Mode

Enable verbose logging for debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Then run your conversion
smart_convert_megatron_to_hf(input_path, output_path)
```

## Examples

See `examples/example_tensor_parallel.py` for comprehensive usage examples.

## API Reference

### TensorParallelConverter

Main class for tensor parallel conversions.

#### Methods

- `convert_minicpm_megatron_to_hf_tp_pp()`: Convert MiniCPM Megatron to HF
- `convert_llama_megatron_to_hf_tp_pp()`: Convert Llama Megatron to HF
- `validate_parallel_config()`: Validate parallel configuration
- `load_distributed_checkpoints()`: Load distributed checkpoints

### SmartConverter

Smart conversion scheduler with auto-detection.

#### Methods

- `convert_megatron_to_hf()`: Smart Megatron to HF conversion
- `convert_hf_to_megatron()`: Smart HF to Megatron conversion
- `detect_model_type_and_size()`: Auto-detect model characteristics
- `detect_parallel_config()`: Auto-detect parallel configuration
- `detect_model_variant()`: Detect model variant (e.g., MiniCPM-4)

### Convenience Functions

- `smart_convert_megatron_to_hf()`: One-click smart conversion
- `smart_convert_hf_to_megatron()`: One-click smart reverse conversion
- `convert_minicpm_8b()`, `convert_minicpm_3b()`: Model-specific conversions
- `convert_llama_7b()`: Llama-specific conversion 