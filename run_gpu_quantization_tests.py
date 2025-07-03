#!/usr/bin/env python3
"""
GPU服务器量化测试脚本
基于现有的test_quantization.py，但移除CPU限制，启用GPU加速
"""

import os
import sys
import time
import torch
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# GPU服务器环境设置 - 移除CPU限制，启用GPU
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 移除这些限制，让GPU可用
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# os.environ["USE_CPU_ONLY"] = "1"

def check_gpu_environment():
    """检查GPU环境"""
    logger.info("=== GPU环境检查 ===")
    
    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"✅ CUDA可用: {device_name}")
        logger.info(f"   GPU内存: {memory_gb:.2f} GB")
        logger.info(f"   CUDA版本: {torch.version.cuda}")
        return "cuda"
    elif mps_available:
        logger.info("✅ MPS可用")
        return "mps"
    else:
        logger.info("❌ 仅CPU可用")
        return "cpu"

def run_quantization_tests():
    """运行量化测试"""
    logger.info("=== 开始GPU量化测试 ===")
    
    from model_converter_tool.converter import ModelConverter
    converter = ModelConverter()
    
    # 测试配置
    test_models = [
        "facebook/opt-125m",  # 小模型，快速测试
        "sshleifer/tiny-gpt2"  # 更小的模型
    ]
    
    output_dir = Path("test_outputs/gpu_quantization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    test_results = {}
    
    # 测试GPTQ量化
    logger.info("测试GPTQ量化...")
    start_time = time.time()
    
    result = converter.convert(
        input_source="facebook/opt-125m",
        output_format="gptq",
        output_path=str(output_dir / "opt_125m_gptq"),
        model_type="text-generation",
        device="auto",  # 自动选择最佳设备
        validate=True,
    )
    
    gptq_duration = time.time() - start_time
    test_results["gptq"] = {
        "success": result["success"],
        "duration": gptq_duration,
        "error": result.get("error"),
        "validation": result.get("model_validation", {})
    }
    
    if result["success"]:
        logger.info(f"✅ GPTQ量化成功，耗时: {gptq_duration:.2f}秒")
    else:
        logger.error(f"❌ GPTQ量化失败: {result.get('error')}")
    
    # 测试AWQ量化
    logger.info("测试AWQ量化...")
    start_time = time.time()
    
    result = converter.convert(
        input_source="facebook/opt-125m",
        output_format="awq",
        output_path=str(output_dir / "opt_125m_awq"),
        model_type="text-generation",
        device="auto",
        validate=True,
    )
    
    awq_duration = time.time() - start_time
    test_results["awq"] = {
        "success": result["success"],
        "duration": awq_duration,
        "error": result.get("error"),
        "validation": result.get("model_validation", {})
    }
    
    if result["success"]:
        logger.info(f"✅ AWQ量化成功，耗时: {awq_duration:.2f}秒")
    else:
        logger.error(f"❌ AWQ量化失败: {result.get('error')}")
    
    # 测试GGUF量化
    logger.info("测试GGUF量化...")
    quantization_levels = ["q4_k_m", "q8_0", "q5_k_m"]
    gguf_results = {}
    
    for quant_level in quantization_levels:
        logger.info(f"  测试GGUF量化级别: {quant_level}")
        start_time = time.time()
        
        result = converter.convert(
            input_source="sshleifer/tiny-gpt2",
            output_format="gguf",
            output_path=str(output_dir / f"tiny_gpt2_gguf_{quant_level}.gguf"),
            model_type="text-generation",
            quantization=quant_level,
            device="auto",
            validate=True,
        )
        
        duration = time.time() - start_time
        gguf_results[quant_level] = {
            "success": result["success"],
            "duration": duration,
            "error": result.get("error"),
            "validation": result.get("model_validation", {})
        }
        
        if result["success"]:
            logger.info(f"    ✅ GGUF {quant_level} 成功，耗时: {duration:.2f}秒")
        else:
            logger.error(f"    ❌ GGUF {quant_level} 失败: {result.get('error')}")
    
    test_results["gguf"] = gguf_results
    
    return test_results

def analyze_results(test_results):
    """分析测试结果"""
    logger.info("=== 测试结果分析 ===")
    
    # 总体统计
    total_tests = 2 + len(test_results["gguf"])  # GPTQ + AWQ + GGUF
    successful_tests = 0
    
    # GPTQ结果
    gptq_result = test_results["gptq"]
    if gptq_result["success"]:
        successful_tests += 1
        logger.info(f"GPTQ量化: ✅ 成功 ({gptq_result['duration']:.2f}秒)")
    else:
        logger.error(f"GPTQ量化: ❌ 失败 - {gptq_result['error']}")
    
    # AWQ结果
    awq_result = test_results["awq"]
    if awq_result["success"]:
        successful_tests += 1
        logger.info(f"AWQ量化: ✅ 成功 ({awq_result['duration']:.2f}秒)")
    else:
        logger.error(f"AWQ量化: ❌ 失败 - {awq_result['error']}")
    
    # GGUF结果
    logger.info("GGUF量化结果:")
    for level, result in test_results["gguf"].items():
        if result["success"]:
            successful_tests += 1
            logger.info(f"  {level}: ✅ 成功 ({result['duration']:.2f}秒)")
        else:
            logger.error(f"  {level}: ❌ 失败 - {result['error']}")
    
    # 成功率
    success_rate = successful_tests / total_tests * 100
    logger.info(f"总体成功率: {success_rate:.1f}% ({successful_tests}/{total_tests})")
    
    if success_rate >= 80:
        logger.info("🎉 GPU量化测试总体成功！")
    elif success_rate >= 50:
        logger.info("⚠️ GPU量化测试部分成功")
    else:
        logger.error("❌ GPU量化测试失败较多")
    
    return success_rate

def main():
    """主函数"""
    logger.info("开始GPU服务器量化测试")
    
    # 1. 检查GPU环境
    device = check_gpu_environment()
    
    # 2. 运行量化测试
    test_results = run_quantization_tests()
    
    # 3. 分析结果
    success_rate = analyze_results(test_results)
    
    # 4. 保存结果
    import json
    with open("gpu_quantization_test_results.json", "w", encoding="utf-8") as f:
        # 移除不可序列化的对象
        serializable_results = {}
        for key, value in test_results.items():
            if isinstance(value, dict):
                serializable_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, (str, int, float, bool, list)):
                        serializable_results[key][k] = v
                    elif isinstance(v, dict):
                        serializable_results[key][k] = {}
                        for k2, v2 in v.items():
                            if isinstance(v2, (str, int, float, bool, list)):
                                serializable_results[key][k][k2] = v2
            elif isinstance(value, (str, int, float, bool)):
                serializable_results[key] = value
        
        serializable_results["device"] = device
        serializable_results["success_rate"] = success_rate
        
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    logger.info("测试完成，结果已保存到 gpu_quantization_test_results.json")
    
    # 返回退出码
    return 0 if success_rate >= 50 else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 