#!/usr/bin/env python3
"""
Auto-install mlx on Apple Silicon macOS
"""

import platform
import subprocess
import sys

def install_mlx_if_needed():
    """Install mlx if on Apple Silicon macOS"""
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        print("🍎 Detected Apple Silicon macOS - installing mlx for optimized inference...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "mlx>=0.0.8"])
            print("✅ mlx installed successfully for Apple Silicon optimization")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Failed to install mlx: {e}")
    else:
        print(f"ℹ️  Platform: {platform.system()} {platform.machine()}")
        print("   MLX not available for this platform - MLX features will be disabled")

if __name__ == "__main__":
    install_mlx_if_needed() 