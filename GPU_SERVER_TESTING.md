# GPU服务器量化测试指南

## 概述

本指南说明如何在GPU服务器上运行量化转换测试，验证GPTQ、AWQ、GGUF等量化格式的真实有效性。

## 测试脚本说明

### 1. 现有测试 vs GPU测试的区别

**现有 `tests/test_quantization.py`**:
- 强制使用CPU（通过环境变量限制）
- 使用小模型进行快速测试
- 完整的pytest测试框架

**GPU服务器测试的优势**:
- 启用GPU加速，显著提升量化速度
- 避免MPS兼容性问题
- 真实的量化效果验证

### 2. 测试脚本选择

#### 方案A: 使用专用GPU测试脚本（推荐）
```bash
python run_gpu_quantization_tests.py
```

**特点**:
- 基于现有测试逻辑
- 移除CPU限制，启用GPU
- 详细的性能分析和结果报告
- 自动保存测试结果到JSON文件

#### 方案B: 使用pytest框架
```bash
# 方法1: 直接运行（需要临时修改环境变量）
pytest tests/test_quantization.py -v -m quantization

# 方法2: 使用提供的shell脚本
./run_gpu_pytest.sh
```

## 在GPU服务器上运行

### 1. 环境准备
```bash
# 确保CUDA环境可用
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"

# 检查GPU信息
nvidia-smi
```

### 2. 运行测试
```bash
# 方法1: 运行专用GPU测试脚本
python run_gpu_quantization_tests.py

# 方法2: 使用shell脚本（包含环境检查）
./run_gpu_pytest.sh
```

### 3. 查看结果
```bash
# 查看测试结果文件
cat gpu_quantization_test_results.json

# 查看输出目录
ls -la test_outputs/gpu_quantization/
```

## 预期结果

### 成功指标
- **GPTQ量化**: 成功生成量化模型文件
- **AWQ量化**: 成功生成量化模型文件  
- **GGUF量化**: 成功生成不同精度的GGUF文件
- **性能提升**: GPU加速比 > 1.0x

### 输出文件
```
test_outputs/gpu_quantization/
├── opt_125m_gptq/          # GPTQ量化结果
├── opt_125m_awq/           # AWQ量化结果
└── tiny_gpt2_gguf_*.gguf/  # GGUF量化结果
```

## 故障排除

### 1. CUDA不可用
```bash
# 检查CUDA安装
nvcc --version
python -c "import torch; print(torch.version.cuda)"

# 检查GPU驱动
nvidia-smi
```

### 2. 内存不足
- 使用更小的测试模型
- 减少批处理大小
- 检查GPU内存使用情况

### 3. 依赖问题
```bash
# 检查gptqmodel安装
python -c "import gptqmodel; print('gptqmodel可用')"

# 重新安装依赖
pip install -r requirements.txt
```

## 性能对比

### CPU vs GPU 性能预期
- **GPTQ量化**: GPU加速比 3-10x
- **AWQ量化**: GPU加速比 3-10x  
- **GGUF量化**: GPU加速比 2-5x

### 量化效果验证
- 模型文件大小减少 50-75%
- 推理速度提升 20-50%
- 内存使用减少 50-75%

## 注意事项

1. **模型大小**: 使用小模型进行测试，避免长时间等待
2. **GPU内存**: 确保有足够的GPU内存进行量化
3. **网络连接**: 首次运行需要下载模型，确保网络稳定
4. **存储空间**: 确保有足够的磁盘空间存储量化结果

## 扩展测试

### 1. 测试更大模型
```python
# 修改测试模型
test_models = [
    "facebook/opt-350m",  # 中等大小模型
    "gpt2-medium"         # GPT-2中等模型
]
```

### 2. 测试不同量化精度
```python
# 添加更多量化级别
quantization_levels = ["q2_k", "q3_k_m", "q4_k_m", "q5_k_m", "q6_k", "q8_0"]
```

### 3. 性能基准测试
```bash
# 运行性能对比测试
python quick_gpu_test.py
``` 