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

if $INSTALL_SYS_DEPS; then
  if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "[INFO] 检测到 macOS，使用 Homebrew 安装依赖..."
    if ! command -v brew >/dev/null 2>&1; then
      echo "[INFO] 未检测到 Homebrew，正在自动安装..."
      /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    brew install git make python cmake
  elif [[ -f /etc/debian_version ]]; then
    echo "[INFO] 检测到 Debian/Ubuntu，使用 apt 安装依赖..."
    sudo apt update
    sudo apt install -y git make python3 cmake
  elif [[ -f /etc/redhat-release ]]; then
    echo "[INFO] 检测到 CentOS/RHEL，使用 yum 安装依赖..."
    sudo yum install -y git make python3 cmake
  else
    echo "[ERROR] 未能自动识别操作系统，请手动安装 git make python3 cmake 后重试。"
    exit 1
  fi
  if ! command -v cmake >/dev/null 2>&1; then
    echo "[WARNING] cmake 已安装但未在 PATH 中。"
    echo "请将 Homebrew 的 bin 路径加入 PATH，例如："
    echo '  export PATH="/opt/homebrew/bin:$PATH"'
    echo "并写入 ~/.zshrc 或 ~/.bash_profile 后 source 一下，或重启终端。"
    echo "参考: https://docs.brew.sh/Homebrew-and-Python"
  else
    echo "[SUCCESS] 系统依赖已安装完成！"
  fi
fi

echo "\n开始安装 Python 依赖...\n"
pip3 install -r requirements.txt
if [ $? -eq 0 ]; then
  echo "\n✅ 依赖安装完成！"
else
  echo "\n⚠️ 某些依赖安装失败（如 auto-gptq/autoawq），请参考 README.md 的常见问题说明。"
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