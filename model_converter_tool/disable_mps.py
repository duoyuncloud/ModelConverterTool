#!/usr/bin/env python3
"""
强制禁用 MPS/CUDA 后端的环境变量设置脚本
在任何其他 import 之前导入此脚本，确保彻底禁用 GPU 后端
"""

import os
import sys


def disable_gpu_backends():
    """强制禁用所有 GPU 后端"""
    # 基础禁用
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # os.environ["MPS_VISIBLE_DEVICES"] = ""
    # os.environ["TRANSFORMERS_NO_MPS"] = "1"
    # os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    # os.environ["USE_CPU_ONLY"] = "1"

    # PyTorch MPS 相关
    # os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    # os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.0"

    # PyTorch CUDA 相关
    # os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:0"

    # 其他可能的 GPU 后端
    # os.environ["PYTORCH_ROCM_ARCH"] = ""
    # os.environ["PYTORCH_CUDA_ARCH_LIST"] = ""

    # 强制设置设备为 CPU
    # os.environ["PYTORCH_DEVICE"] = "cpu"

    # print("🔒 GPU backends disabled (MPS/CUDA/ROCm)")


# 自动执行
if __name__ == "__main__":
    disable_gpu_backends()
else:
    # 作为模块导入时也自动执行
    disable_gpu_backends()
