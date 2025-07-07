#!/bin/bash

echo "==============================="
echo "Model Converter Tool 安装依赖"
echo "==============================="

# 检测操作系统
if [[ "$OSTYPE" == "darwin"* ]]; then
  echo "\n⚠️ auto-gptq/autoawq 仅支持 Linux + NVIDIA 显卡环境，macOS 下可忽略安装报错"
fi

if [[ "$OSTYPE" == "linux"* ]]; then
  echo "\n如需 GPTQ/AWQ 真实量化，请确保已安装 NVIDIA 显卡驱动和 CUDA 环境。"
fi

echo "\n开始安装依赖...\n"
pip install -r requirements.txt

if [ $? -eq 0 ]; then
  echo "\n✅ 依赖安装完成！"
else
  echo "\n⚠️ 某些依赖安装失败（如 auto-gptq/autoawq），请参考 README.md 的常见问题说明。"
fi

# Apple Silicon (macOS arm64) 下自动安装 mlx
python install_mlx.py

if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "x86_64" ]]; then
    echo "ℹ️  Detected Intel macOS - mlx is not available for Intel Macs"
    echo "   MLX features will be disabled. Consider using Apple Silicon for MLX support."
elif [[ "$(uname -s)" == "Linux" ]]; then
    echo "🐧 Detected Linux - mlx is not available for Linux"
    echo "   MLX features will be disabled. MLX only supports Apple Silicon macOS."
else
    echo "🖥️  Detected $(uname -s) $(uname -m) - mlx is not available for this platform"
    echo "   MLX features will be disabled. MLX only supports Apple Silicon macOS."
fi 