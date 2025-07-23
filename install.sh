#!/bin/bash

# Exit on error
set -e

# Check for active virtual environment
if [ ! -z "$VIRTUAL_ENV" ]; then
  echo "Error: Please deactivate any active Python virtual environment before running this script."
  echo "Detected: $VIRTUAL_ENV. Run 'deactivate' and try again."
  exit 1
fi

echo "=== ModelConverterTool Installation ==="

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
RECOMMENDED="3.8 3.9 3.10 3.11"
if [[ ! " $RECOMMENDED " =~ " $PYTHON_VERSION " ]]; then
  echo "Warning: Your Python version is $PYTHON_VERSION. Recommended: 3.8~3.11."
fi

# Ensure python3.11 is available
if ! command -v python3.11 &>/dev/null; then
  echo "Error: python3.11 is required but not found. Please install Python 3.11."
  exit 1
fi

# Remove old venv if not using python3.11
if [ -d "venv" ]; then
  VENV_PY=$(venv/bin/python --version 2>/dev/null | awk '{print $2}')
  if [[ $VENV_PY != 3.11* ]]; then
    echo "Removing old venv (Python $VENV_PY) to recreate with Python 3.11..."
    rm -rf venv
  fi
fi

# Create virtual environment if needed
if [ ! -d "venv" ]; then
  python3.11 -m venv venv
  echo "Virtual environment created at ./venv (Python 3.11)"
fi

# Verify venv Python version
VENV_PY=$(venv/bin/python --version 2>/dev/null | awk '{print $2}')
if [[ $VENV_PY != 3.11* ]]; then
  echo "Error: venv Python version is $VENV_PY, expected 3.11.x. Aborting."
  exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip, setuptools, wheel
pip install --upgrade pip setuptools wheel

# Install PyTorch
pip install torch torchvision torchaudio

# Install other dependencies
pip install -U -r requirements.txt

# Install llama.cpp requirements if present
if [ -f "llama.cpp/requirements.txt" ]; then
  pip install --no-cache-dir -r llama.cpp/requirements.txt
fi

# macOS: Install Xcode Command Line Tools if needed
if ! xcode-select -p &>/dev/null; then
  echo "Installing Xcode Command Line Tools..."
  xcode-select --install
fi

# macOS: Install CMake if needed
if ! command -v cmake &>/dev/null; then
  echo "Installing CMake via Homebrew..."
  if command -v brew &>/dev/null; then
    brew install cmake
  else
    echo "Homebrew not found. Please install Homebrew first:"
    echo '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    exit 1
  fi
fi

# Install package as editable CLI tool
pip install -e .

echo ""
echo "=== Installation complete! ==="
echo "To activate your environment, run:"
echo "  source venv/bin/activate"
echo "Always use the virtual environment for best results."