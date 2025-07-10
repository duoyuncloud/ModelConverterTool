#!/bin/bash

set -e

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
  python3 -m venv venv
  echo "Virtual environment created. Run 'source venv/bin/activate' to activate."
fi

# Activate venv
source venv/bin/activate

# Install in editable mode
pip install -e .

# 自动安装 llama.cpp 的 Python 依赖
if [ -f "llama.cpp/requirements.txt" ]; then
  pip install --no-cache-dir -r llama.cpp/requirements.txt
elif [ -d "llama.cpp/requirements" ]; then
  for req in llama.cpp/requirements/*.txt; do
    pip install --no-cache-dir -r "$req"
  done
else
  echo "Warning: llama.cpp requirements not found, please check your repo."
fi

echo "\nInstallation complete!"
echo "Activate your environment with: source venv/bin/activate"
echo "You can now use the CLI: modelconvert --help" 