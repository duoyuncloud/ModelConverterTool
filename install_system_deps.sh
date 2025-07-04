#!/bin/bash
set -e

# 检测操作系统
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

# 检查 cmake 是否在 PATH
if ! command -v cmake >/dev/null 2>&1; then
  echo "[WARNING] cmake 已安装但未在 PATH 中。"
  echo "请将 Homebrew 的 bin 路径加入 PATH，例如："
  echo '  export PATH="/opt/homebrew/bin:$PATH"' 
  echo "并写入 ~/.zshrc 或 ~/.bash_profile 后 source 一下，或重启终端。"
  echo "参考: https://docs.brew.sh/Homebrew-and-Python"
else
  echo "[SUCCESS] 系统依赖已安装完成！"
fi 