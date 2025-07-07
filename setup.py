#!/usr/bin/env python3
"""
Dynamic setup.py for model_converter_tool
Automatically detects Apple Silicon and adds mlx dependency
"""

import platform
import sys
from setuptools import setup

def get_dependencies():
    """Get dependencies based on platform"""
    base_deps = [
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.21.0",
        "typer>=0.9.0",
        "rich>=13.0.0",
        "pyyaml>=6.0",
        "requests>=2.28.0",
        "tqdm>=4.64.0",
        "onnx>=1.14.0",
        "onnxruntime>=1.15.0",
        "safetensors>=0.3.0",
        # 量化相关依赖
        "gptqmodel>=0.1.0",
    ]
    
    # Check if we're on Apple Silicon macOS
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        print("🍎 Detected Apple Silicon macOS - adding mlx dependency for optimized inference")
        base_deps.append("mlx>=0.0.8")
    else:
        print(f"ℹ️  Platform: {platform.system()} {platform.machine()}")
        print("   MLX not available for this platform - MLX features will be disabled")
    
    return base_deps

if __name__ == "__main__":
    setup(
        install_requires=get_dependencies(),
    )
