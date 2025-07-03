#!/usr/bin/env python3
"""
快速GPU量化测试脚本
验证GPU服务器下的量化转换是否真实有效
"""

import os
import time
import torch
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 强制使用CPU进行量化（避免MPS兼容性问题）
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["USE_CPU_ONLY"] = "1"

def check_gpu():
    """检查GPU状态"""
    logger.info("=== GPU状态检查 ===")
    
    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"✅ CUDA可用: {device_name}")
        logger.info(f"   GPU内存: {memory_gb:.2f} GB")
        return "cuda"
    elif mps_available:
        logger.info("✅ MPS可用（但量化测试将使用CPU）")
        return "mps"
    else:
        logger.info("❌ 仅CPU可用")
        return "cpu"

def test_gpu_quantization():
    """测试GPU量化"""
    logger.info("=== 量化测试（使用CPU避免兼容性问题） ===")
    
    # 导入转换器
    from model_converter_tool.converter import ModelConverter
    converter = ModelConverter()
    
    # 测试模型
    test_model = "facebook/opt-125m"
    output_dir = Path("test_outputs/quick_gpu_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 测试GPTQ量化
    logger.info("测试GPTQ量化...")
    start_time = time.time()
    
    result = converter.convert(
        input_source=test_model,
        output_format="gptq",
        output_path=str(output_dir / "gptq_test"),
        model_type="text-generation",
        device="cpu",  # 强制使用CPU
        validate=True
    )
    
    duration = time.time() - start_time
    
    if result["success"]:
        logger.info(f"✅ GPTQ量化成功，耗时: {duration:.2f}秒")
        
        # 检查输出文件
        output_path = output_dir / "gptq_test"
        if output_path.exists():
            files = [f.name for f in output_path.iterdir() if f.is_file()]
            logger.info(f"   输出文件: {files}")
            
            # 检查是否有量化文件
            has_quantized_files = any(f.endswith(('.safetensors', '.bin')) for f in files)
            if has_quantized_files:
                logger.info("✅ 发现量化模型文件")
            else:
                logger.warning("⚠️ 未发现量化模型文件")
        else:
            logger.error("❌ 输出目录不存在")
    else:
        logger.error(f"❌ GPTQ量化失败: {result.get('error')}")
    
    return result["success"], duration

def test_awq_quantization():
    """测试AWQ量化"""
    logger.info("=== AWQ量化测试 ===")
    
    from model_converter_tool.converter import ModelConverter
    converter = ModelConverter()
    
    test_model = "facebook/opt-125m"
    output_dir = Path("test_outputs/quick_gpu_test")
    
    logger.info("测试AWQ量化...")
    start_time = time.time()
    
    result = converter.convert(
        input_source=test_model,
        output_format="awq",
        output_path=str(output_dir / "awq_test"),
        model_type="text-generation",
        device="cpu",  # 强制使用CPU
        validate=True
    )
    
    duration = time.time() - start_time
    
    if result["success"]:
        logger.info(f"✅ AWQ量化成功，耗时: {duration:.2f}秒")
        
        output_path = output_dir / "awq_test"
        if output_path.exists():
            files = [f.name for f in output_path.iterdir() if f.is_file()]
            logger.info(f"   输出文件: {files}")
    else:
        logger.error(f"❌ AWQ量化失败: {result.get('error')}")
    
    return result["success"], duration

def test_gguf_quantization():
    """测试GGUF量化"""
    logger.info("=== GGUF量化测试 ===")
    
    from model_converter_tool.converter import ModelConverter
    converter = ModelConverter()
    
    test_model = "facebook/opt-125m"
    output_dir = Path("test_outputs/quick_gpu_test")
    
    quantization_levels = ["q4_k_m", "q8_0"]
    results = {}
    
    for quant_level in quantization_levels:
        logger.info(f"测试GGUF量化级别: {quant_level}")
        start_time = time.time()
        
        result = converter.convert(
            input_source=test_model,
            output_format="gguf",
            output_path=str(output_dir / f"gguf_{quant_level}.gguf"),
            model_type="text-generation",
            quantization=quant_level,
            device="cpu",  # 强制使用CPU
            validate=True
        )
        
        duration = time.time() - start_time
        
        if result["success"]:
            # 检查GGUF文件
            gguf_dir = output_dir / f"gguf_{quant_level}.gguf"
            gguf_file = None
            
            # 查找.gguf文件
            for file_path in gguf_dir.iterdir():
                if file_path.is_file() and file_path.suffix == '.gguf':
                    gguf_file = file_path
                    break
            
            if gguf_file and gguf_file.exists():
                file_size_mb = gguf_file.stat().st_size / 1024**2
                logger.info(f"✅ GGUF量化 {quant_level} 成功，耗时: {duration:.2f}秒，文件大小: {file_size_mb:.2f}MB")
                results[quant_level] = {"success": True, "duration": duration, "file_size_mb": file_size_mb}
            else:
                logger.error(f"❌ GGUF量化 {quant_level} 成功但输出文件不存在")
                results[quant_level] = {"success": False, "error": "输出文件不存在"}
        else:
            logger.error(f"❌ GGUF量化 {quant_level} 失败: {result.get('error')}")
            results[quant_level] = {"success": False, "error": result.get('error')}
    
    return results

def test_cpu_vs_gpu():
    """对比CPU和GPU性能"""
    logger.info("=== CPU vs GPU性能对比 ===")
    
    from model_converter_tool.converter import ModelConverter
    converter = ModelConverter()
    
    test_model = "facebook/opt-125m"
    output_dir = Path("test_outputs/quick_gpu_test")
    
    # CPU测试
    logger.info("CPU量化测试...")
    start_time = time.time()
    
    cpu_result = converter.convert(
        input_source=test_model,
        output_format="gptq",
        output_path=str(output_dir / "cpu_test"),
        model_type="text-generation",
        device="cpu",
        validate=False
    )
    cpu_time = time.time() - start_time
    
    # GPU测试（如果有CUDA）
    if torch.cuda.is_available():
        logger.info("GPU量化测试...")
        start_time = time.time()
        
        gpu_result = converter.convert(
            input_source=test_model,
            output_format="gptq",
            output_path=str(output_dir / "gpu_test"),
            model_type="text-generation",
            device="cuda",
            validate=False
        )
        gpu_time = time.time() - start_time
        
        if cpu_result["success"] and gpu_result["success"]:
            speedup = cpu_time / gpu_time
            logger.info(f"CPU耗时: {cpu_time:.2f}秒")
            logger.info(f"GPU耗时: {gpu_time:.2f}秒")
            logger.info(f"GPU加速比: {speedup:.2f}x")
            
            if speedup > 1.0:
                logger.info("✅ GPU确实提供了加速")
            else:
                logger.warning("⚠️ GPU未提供加速")
        else:
            logger.error("❌ 性能对比测试失败")
            speedup = None
    else:
        logger.info("无CUDA GPU，跳过GPU性能对比")
        speedup = None
    
    return cpu_result["success"], speedup, cpu_time

def test_quantization_effectiveness():
    """测试量化效果"""
    logger.info("=== 量化效果测试 ===")
    
    output_dir = Path("test_outputs/quick_gpu_test")
    
    # 检查量化模型文件大小
    gptq_path = output_dir / "gptq_test"
    if gptq_path.exists():
        total_size = 0
        file_count = 0
        
        for file_path in gptq_path.iterdir():
            if file_path.is_file():
                size_mb = file_path.stat().st_size / 1024**2
                total_size += size_mb
                file_count += 1
                logger.info(f"文件: {file_path.name}, 大小: {size_mb:.2f}MB")
        
        logger.info(f"量化模型总大小: {total_size:.2f}MB ({file_count}个文件)")
        
        # 估算原始模型大小（OPT-125M大约500MB）
        original_size = 500  # MB
        compression_ratio = original_size / total_size if total_size > 0 else 0
        
        logger.info(f"压缩比: {compression_ratio:.2f}x")
        
        if compression_ratio > 1.5:
            logger.info("✅ 量化压缩效果明显")
        else:
            logger.warning("⚠️ 量化压缩效果不明显")
    else:
        logger.error("❌ 量化模型目录不存在")

def main():
    """主函数"""
    logger.info("开始快速量化测试")
    
    # 1. 检查GPU
    device = check_gpu()
    
    # 2. 测试GPTQ量化
    gptq_success, gptq_duration = test_gpu_quantization()
    
    # 3. 测试AWQ量化
    awq_success, awq_duration = test_awq_quantization()
    
    # 4. 测试GGUF量化
    gguf_results = test_gguf_quantization()
    
    # 5. 性能对比
    cpu_success, speedup, cpu_time = test_cpu_vs_gpu()
    
    # 6. 量化效果
    test_quantization_effectiveness()
    
    # 总结
    logger.info("=== 测试总结 ===")
    logger.info(f"设备: {device}")
    logger.info(f"GPTQ量化: {'✅ 成功' if gptq_success else '❌ 失败'}")
    if gptq_success:
        logger.info(f"  - 耗时: {gptq_duration:.2f}秒")
    
    logger.info(f"AWQ量化: {'✅ 成功' if awq_success else '❌ 失败'}")
    if awq_success:
        logger.info(f"  - 耗时: {awq_duration:.2f}秒")
    
    logger.info("GGUF量化结果:")
    for level, result in gguf_results.items():
        status = "✅ 成功" if result.get('success') else "❌ 失败"
        logger.info(f"  - {level}: {status}")
        if result.get('success'):
            logger.info(f"    - 耗时: {result['duration']:.2f}秒")
            if 'file_size_mb' in result:
                logger.info(f"    - 文件大小: {result['file_size_mb']:.2f}MB")
    
    if speedup:
        logger.info(f"GPU加速比: {speedup:.2f}x")
        
        if speedup > 1.0:
            logger.info("🎉 GPU量化转换真实有效！")
        else:
            logger.info("⚠️ GPU量化转换效果不明显")
    else:
        logger.info("ℹ️ 无法进行GPU性能对比")
    
    # 总体成功率
    total_tests = 1 + 1 + len(gguf_results)  # GPTQ + AWQ + GGUF
    successful_tests = sum([gptq_success, awq_success] + [r.get('success', False) for r in gguf_results.values()])
    success_rate = successful_tests / total_tests * 100
    
    logger.info(f"总体成功率: {success_rate:.1f}% ({successful_tests}/{total_tests})")

if __name__ == "__main__":
    main() 