"""
Batch example: Convert multiple models using a YAML configuration file.
"""

import os

# The config file should define a list of conversion tasks (see configs/batch_template.yaml)
config_path = "../configs/batch_template.yaml"

# Run batch conversion using the config file
os.system(f"python -m model_converter_tool.cli batch {config_path}")
