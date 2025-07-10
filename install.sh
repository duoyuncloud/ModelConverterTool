#!/bin/bash

set -e

show_help() {
  echo "Usage: $0 [--system-deps] [--enable-mlx]"
  echo "  --system-deps   Install system dependencies (git, make, python, cmake)"
  echo "  --enable-mlx    Install mlx for Apple Silicon macOS"
  exit 1
}

INSTALL_SYS_DEPS=false
ENABLE_MLX=false

for arg in "$@"; do
  case $arg in
    --system-deps)
      INSTALL_SYS_DEPS=true
      ;;
    --enable-mlx)
      ENABLE_MLX=true
      ;;
    --help|-h)
      show_help
      ;;
    *)
      echo "Unknown argument: $arg"
      show_help
      ;;
  esac
done

# Check for virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
  echo "[ERROR] You are not running inside a Python virtual environment."
  echo "Please create and activate a venv first:"
  echo "  python3 -m venv venv"
  echo "  source venv/bin/activate"
  echo "Then re-run this script."
  exit 1
fi

# Ensure torch is installed before other requirements
if ! python3 -c "import torch" 2>/dev/null; then
  echo "[INFO] torch not found, installing torch first..."
  pip3 install torch
else
  echo "[INFO] torch is already installed."
fi

echo -e "\nInstalling Python dependencies...\n"
pip3 install -r requirements.txt
if [ $? -eq 0 ]; then
  echo -e "\n✅ Dependencies installed successfully!"
else
  echo -e "\n⚠️ Some dependencies failed to install (e.g., auto-gptq/autoawq). Please see the FAQ in README.md."
fi

if $ENABLE_MLX; then
  if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    echo "🍎 Detected Apple Silicon macOS - installing mlx for optimized inference..."
    python3 -m pip install mlx>=0.0.8
    if [ $? -eq 0 ]; then
      echo "✅ mlx installed successfully for Apple Silicon optimization"
    else
      echo "⚠️  Failed to install mlx"
    fi
  else
    echo "MLX only supports Apple Silicon macOS. Skipping mlx installation."
  fi
fi 