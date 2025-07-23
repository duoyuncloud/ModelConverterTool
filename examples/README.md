# Examples

This directory contains example scripts demonstrating typical usage of the Model Converter Tool.

## Available Examples

- `example_basic.py`: Basic model conversion using the CLI.
- `example_batch.py`: Batch conversion using a YAML config file.
- `example_quantization.py`: Model quantization during conversion.
- `example_api.py`: Using the tool programmatically via the Python API.
- `example_mup2llama.py`: Convert a muP-initialized model to LLaMA format using `--mup2llama`.
- `example_fake_weight.py`: Use `--fake-weight` and `--fake-weight-config` to generate models with zero or custom-shaped weights.

## How to Run

```bash
cd examples
python example_basic.py
python example_batch.py
python example_quantization.py
python example_api.py
python example_mup2llama.py
python example_fake_weight.py
```

See each script for details. For advanced features, refer to the CLI help or API documentation. 