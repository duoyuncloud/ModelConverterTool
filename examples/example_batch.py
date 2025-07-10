"""
Batch example: Convert multiple models using a YAML configuration file.
"""
import os

# Resolve the absolute path to the config file
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(project_root, "configs", "batch_template.yaml")

# Example command: batch conversion using a config file
os.system(
    f"python -m model_converter_tool.cli batch {config_path}"
) 