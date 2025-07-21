"""
Fake weight example: Generate a model with zero or custom-shaped weights for testing.
"""

import os

# Example command: convert a model with all weights set to zero (for structure/debug)
os.system(
    "python -m model_converter_tool.cli convert path/to/input_model --to safetensors --output path/to/fake_model --fake-weight"
)

# Example command: use a custom fake weight config (YAML or JSON) for specific shapes
# os.system(
#     "python -m model_converter_tool.cli convert path/to/input_model --to safetensors --output path/to/fake_model_custom --fake-weight --fake-weight-config path/to/fake_weight_config.yaml"
# )
