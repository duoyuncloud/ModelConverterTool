#!/bin/bash

# Tsinghua PyPI mirror URL (set once at the top)
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"

# Check if running inside a virtual environment
if [ ! -z "$VIRTUAL_ENV" ]; then
  echo "Error: Please deactivate any active Python virtual environment before running this script."
  echo "Detected that you are in a virtual environment: $VIRTUAL_ENV. Please run 'deactivate' and then execute ./install.sh again."
  exit 1
fi

set -e

echo "=== ModelConverterTool Installation ==="

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
RECOMMENDED="3.8 3.9 3.10 3.11"
if [[ ! " $RECOMMENDED " =~ " $PYTHON_VERSION " ]]; then
  echo "Warning: Your Python version is $PYTHON_VERSION. Recommended: 3.8~3.11 for best compatibility."
fi

# Check for python3.11 availability
if ! command -v python3.11 &>/dev/null; then
  echo "Error: python3.11 is required but not found. Please install Python 3.11 and try again."
  exit 1
fi

# Remove existing venv if it exists and is not using python3.11
if [ -d "venv" ]; then
  VENV_PY=$(venv/bin/python --version 2>/dev/null | awk '{print $2}')
  if [[ $VENV_PY != 3.11* ]]; then
    echo "Existing venv uses Python $VENV_PY, removing it to recreate with Python 3.11..."
    rm -rf venv
  fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
  python3.11 -m venv venv
  echo "Virtual environment created at ./venv using python3.11"
fi

# Verify virtual environment Python version
VENV_PY=$(venv/bin/python --version 2>/dev/null | awk '{print $2}')
if [[ $VENV_PY != 3.11* ]]; then
  echo "Error: venv Python version is $VENV_PY, expected 3.11.x. Aborting."
  exit 1
fi

# ----------------------
# Check Tsinghua PyPI mirror accessibility
# ----------------------
echo ""
echo "Checking Tsinghua PyPI mirror accessibility..."
status_code=$(curl -s -o /dev/null -w "%{http_code}" $PIP_MIRROR)
if [[ $status_code =~ ^(2|3) ]]; then
  echo "✅ Mirror is reachable with status code $status_code. Using: $PIP_MIRROR"
  PIP_INDEX="-i $PIP_MIRROR"
else
  echo "⚠️  Warning: Cannot reach $PIP_MIRROR (status code $status_code). Falling back to default PyPI."
  PIP_INDEX=""
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip, setuptools and wheel first
pip install --upgrade pip setuptools wheel $PIP_INDEX

# Install PyTorch first
pip install torch torchvision torchaudio $PIP_INDEX

# Install/upgrade other dependencies
pip install -U -r requirements.txt $PIP_INDEX

# Optionally install extra requirements for submodules
if [ -f "llama.cpp/requirements.txt" ]; then
  pip install --no-cache-dir -r llama.cpp/requirements.txt $PIP_INDEX
fi

# Check and install Xcode Command Line Tools (macOS only)
if ! xcode-select -p &>/dev/null; then
  echo "Installing Xcode Command Line Tools..."
  xcode-select --install
fi

# Check and install CMake (via Homebrew on macOS)
if ! command -v cmake &>/dev/null; then
  echo "Installing CMake via Homebrew..."
  if command -v brew &>/dev/null; then
    brew install cmake
  else
    echo "Homebrew not found. Please install Homebrew first by running:"
    echo '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    exit 1
  fi
fi

# Install the local package as an editable CLI tool
pip install -e . $PIP_INDEX

echo ""
echo "=== Installation complete! ==="
echo "To activate your environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "For best results, always use the virtual environment!"
