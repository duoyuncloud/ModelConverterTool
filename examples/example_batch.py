"""
Batch example: Convert multiple models using a YAML configuration file.
"""

import os

# Use the provided batch template config
config_path = "../configs/batch_template.yaml"

# Run batch conversion
os.system(f"python -m model_converter_tool.cli batch {config_path}")
