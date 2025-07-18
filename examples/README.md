# Examples

This directory contains Python example scripts demonstrating typical usage patterns of the Model Converter Tool.

Each script is self-contained and can be run directly after installing the required dependencies.

## Available Examples

- `example_basic.py`: Basic model conversion from one format to another using the CLI.
- `example_batch.py`: Batch conversion using a YAML configuration file.
- `example_quantization.py`: Model quantization during conversion.
- `example_api.py`: Using the Model Converter Tool programmatically via its Python API.
- `example_mup2llama.py` (recommended to add): Convert a muP-initialized model to LLaMA format using `--mup2llama` for scaling and config adaptation.
- `example_fake_weight.py` (recommended to add): Use `--fake-weight` and `--fake-weight-config` to generate models with zero or custom-shaped weights for testing.

## How to Run

```bash
cd examples
python example_basic.py
python example_batch.py
python example_quantization.py
python example_api.py
# python example_mup2llama.py   # Add this for muP-to-LLaMA conversion
# python example_fake_weight.py # Add this for fake weight generation
```

See each script for detailed usage and comments. For advanced features, refer to the CLI help or API documentation. 