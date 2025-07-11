#!/bin/bash

set -e

echo "=== ModelConverterTool Installation ==="

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
RECOMMENDED="3.8 3.9 3.10 3.11"
if [[ ! " $RECOMMENDED " =~ " $PYTHON_VERSION " ]]; then
  echo "Warning: Your Python version is $PYTHON_VERSION. Recommended: 3.8~3.11 for best compatibility."
fi

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
  python3 -m venv venv
  echo "Virtual environment created at ./venv"
fi

# Activate venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install/upgrade dependencies
pip install -U -r requirements.txt

# Optionally, install extra requirements for submodules
if [ -f "llama.cpp/requirements.txt" ]; then
  pip install --no-cache-dir -r llama.cpp/requirements.txt
fi

echo ""
echo "=== Installation complete! ==="
echo "To activate your environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "For best results, always use the virtual environment!" 