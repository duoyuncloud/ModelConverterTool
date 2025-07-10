#!/bin/bash

set -e

# Python version check (must be < 3.12 for best compatibility)
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ $(echo "$PYTHON_VERSION >= 3.12" | bc) -eq 1 ]]; then
  echo -e "\033[31m[ERROR]\033[0m Python $PYTHON_VERSION detected. Some dependencies (e.g., sentencepiece) may not install on Python 3.12 or newer."
  echo "Please use Python 3.10 or 3.11 for best compatibility."
  echo "Create a new venv with:"
  echo "  python3.11 -m venv venv"
  echo "  source venv/bin/activate"
  echo "Then re-run this script."
  exit 1
fi

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

# System dependency check for cmake and pkg-config
missing_sys_deps=""
if ! command -v cmake >/dev/null 2>&1; then
  missing_sys_deps="cmake"
fi
if ! command -v pkg-config >/dev/null 2>&1; then
  missing_sys_deps="$missing_sys_deps pkg-config"
fi
if [ -n "$missing_sys_deps" ]; then
  echo -e "\033[31m[ERROR]\033[0m Missing required system dependencies: $missing_sys_deps"
  echo "Please install them first. On macOS, run:"
  echo "  brew install$missing_sys_deps"
  echo "Or use your system's package manager."
  echo "Then re-run this script."
  exit 1
fi

# Check for virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
  echo -e "\033[31m[ERROR]\033[0m You are not running inside a Python virtual environment."
  echo "Please create and activate a venv first:"
  echo "  python3 -m venv venv"
  echo "  source venv/bin/activate"
  echo "Then re-run this script."
  exit 1
fi

# User-friendly torch install
if ! python3 -c "import torch" 2>/dev/null; then
  echo -e "\n[INFO] Installing torch first (required for some packages)...\n"
  if pip3 install torch; then
    echo -e "\033[32m[OK]\033[0m torch installed successfully."
  else
    echo -e "\033[31m[ERROR]\033[0m Failed to install torch. Please check your internet connection or install torch manually."
    exit 1
  fi
else
  echo -e "[INFO] torch is already installed."
fi

echo -e "\n[INFO] Installing remaining Python dependencies...\n"
if pip3 install -r requirements.txt; then
  echo -e "\033[32m[OK]\033[0m All dependencies installed successfully!"
else
  echo -e "\033[33m[WARNING]\033[0m Some dependencies failed to install. See the error above."
  # Automated workaround for gptqmodel/auto-gptq
  if grep -qE '^gptqmodel' requirements.txt; then
    echo -e "\n[INFO] Retrying gptqmodel install with --no-build-isolation..."
    if pip3 install --no-build-isolation gptqmodel; then
      echo -e "\033[32m[OK]\033[0m gptqmodel installed successfully with --no-build-isolation."
    else
      echo -e "\033[31m[ERROR]\033[0m Failed to install gptqmodel even with --no-build-isolation. See the FAQ in README.md."
    fi
  fi
  if grep -qE '^auto-gptq' requirements.txt; then
    echo -e "\n[INFO] Retrying auto-gptq install with --no-build-isolation..."
    if pip3 install --no-build-isolation auto-gptq; then
      echo -e "\033[32m[OK]\033[0m auto-gptq installed successfully with --no-build-isolation."
    else
      echo -e "\033[31m[ERROR]\033[0m Failed to install auto-gptq even with --no-build-isolation. See the FAQ in README.md."
    fi
  fi
fi

if $ENABLE_MLX; then
  if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    echo "🍎 Detected Apple Silicon macOS - installing mlx for optimized inference..."
    if python3 -m pip install mlx>=0.0.8; then
      echo "✅ mlx installed successfully for Apple Silicon optimization"
    else
      echo "⚠️  Failed to install mlx"
    fi
  else
    echo "MLX only supports Apple Silicon macOS. Skipping mlx installation."
  fi
fi

echo -e "\n\033[32m[Done]\033[0m Setup complete!"
echo "To get started:"
echo "  source venv/bin/activate"
echo "  modelconvert --help"
echo "See the README.md for more info." 