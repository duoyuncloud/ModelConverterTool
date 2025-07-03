#!/bin/bash

# GPU服务器量化测试运行脚本
# 基于现有的test_quantization.py，但启用GPU加速

echo "=== GPU服务器量化测试 ==="
echo "时间: $(date)"
echo ""

# 检查CUDA环境
echo "检查CUDA环境..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'✅ CUDA可用: {torch.cuda.get_device_name(0)}')
    print(f'   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
    print(f'   CUDA版本: {torch.version.cuda}')
else:
    print('❌ CUDA不可用')
"

echo ""
echo "运行量化测试..."

# 方法1: 直接运行我们的GPU测试脚本
echo "方法1: 运行GPU专用测试脚本..."
python run_gpu_quantization_tests.py

echo ""
echo "方法2: 运行pytest量化测试（移除CPU限制）..."

# 临时修改环境变量，让pytest使用GPU
export CUDA_VISIBLE_DEVICES=""
export USE_CPU_ONLY=""

# 运行pytest量化测试
pytest tests/test_quantization.py -v -m quantization --tb=short

echo ""
echo "测试完成！"
echo "时间: $(date)" 