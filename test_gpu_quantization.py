#!/usr/bin/env python3
"""
GPU量化转换测试脚本
测试GPU服务器下的量化转换是否真实有效
"""

import os
import sys
import time
import torch
import logging
from pathlib import Path
from typing import Dict, Any

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入转换器
from model_converter_tool.converter import ModelConverter

class GPUQuantizationTester:
    """GPU量化转换测试器"""
    
    def __init__(self):
        self.converter = ModelConverter()
        self.test_model = "facebook/opt-125m"  # 使用较小的模型进行快速测试
        self.output_dir = Path("test_outputs/gpu_quantization")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def check_gpu_availability(self) -> Dict[str, Any]:
        """检查GPU可用性"""
        logger.info("=== GPU可用性检查 ===")
        
        gpu_info = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
            "current_device": "cpu"
        }
        
        if gpu_info["cuda_available"]:
            gpu_info["current_device"] = "cuda"
            gpu_info["cuda_device_name"] = torch.cuda.get_device_name(0)
            gpu_info["cuda_memory_total"] = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            logger.info(f"CUDA可用: {gpu_info['cuda_device_name']}")
            logger.info(f"GPU内存: {gpu_info['cuda_memory_total']:.2f} GB")
        elif gpu_info["mps_available"]:
            gpu_info["current_device"] = "mps"
            logger.info("MPS可用")
        else:
            logger.info("仅CPU可用")
            
        return gpu_info
    
    def test_device_detection(self) -> str:
        """测试设备自动检测"""
        logger.info("=== 设备自动检测测试 ===")
        
        # 测试转换器的设备检测
        result = self.converter.convert(
            input_source=self.test_model,
            output_format="hf",  # 使用简单的HF格式测试
            output_path=str(self.output_dir / "device_test"),
            model_type="text-generation",
            device="auto",
            validate=False
        )
        
        if result["success"]:
            logger.info(f"设备检测成功: {result.get('device_used', 'unknown')}")
            return result.get('device_used', 'unknown')
        else:
            logger.error(f"设备检测失败: {result.get('error')}")
            return "cpu"
    
    def test_gptq_quantization_gpu(self) -> Dict[str, Any]:
        """测试GPU下的GPTQ量化"""
        logger.info("=== GPU GPTQ量化测试 ===")
        
        start_time = time.time()
        
        result = self.converter.convert(
            input_source=self.test_model,
            output_format="gptq",
            output_path=str(self.output_dir / "gptq_gpu"),
            model_type="text-generation",
            device="auto",
            validate=True
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        test_result = {
            "success": result["success"],
            "duration": duration,
            "error": result.get("error"),
            "model_validation": result.get("model_validation", {}),
            "output_files": []
        }
        
        if result["success"]:
            output_path = Path(self.output_dir / "gptq_gpu")
            if output_path.exists():
                test_result["output_files"] = [f.name for f in output_path.iterdir() if f.is_file()]
                logger.info(f"GPTQ量化成功，耗时: {duration:.2f}秒")
                logger.info(f"输出文件: {test_result['output_files']}")
            else:
                logger.error("GPTQ量化成功但输出目录不存在")
        else:
            logger.error(f"GPTQ量化失败: {result.get('error')}")
            
        return test_result
    
    def test_awq_quantization_gpu(self) -> Dict[str, Any]:
        """测试GPU下的AWQ量化"""
        logger.info("=== GPU AWQ量化测试 ===")
        
        start_time = time.time()
        
        result = self.converter.convert(
            input_source=self.test_model,
            output_format="awq",
            output_path=str(self.output_dir / "awq_gpu"),
            model_type="text-generation",
            device="auto",
            validate=True
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        test_result = {
            "success": result["success"],
            "duration": duration,
            "error": result.get("error"),
            "model_validation": result.get("model_validation", {}),
            "output_files": []
        }
        
        if result["success"]:
            output_path = Path(self.output_dir / "awq_gpu")
            if output_path.exists():
                test_result["output_files"] = [f.name for f in output_path.iterdir() if f.is_file()]
                logger.info(f"AWQ量化成功，耗时: {duration:.2f}秒")
                logger.info(f"输出文件: {test_result['output_files']}")
            else:
                logger.error("AWQ量化成功但输出目录不存在")
        else:
            logger.error(f"AWQ量化失败: {result.get('error')}")
            
        return test_result
    
    def test_gguf_quantization_gpu(self) -> Dict[str, Any]:
        """测试GPU下的GGUF量化"""
        logger.info("=== GPU GGUF量化测试 ===")
        
        quantization_levels = ["q4_k_m", "q8_0"]
        results = {}
        
        for quant_level in quantization_levels:
            logger.info(f"测试GGUF量化级别: {quant_level}")
            start_time = time.time()
            
            result = self.converter.convert(
                input_source=self.test_model,
                output_format="gguf",
                output_path=str(self.output_dir / f"gguf_gpu_{quant_level}.gguf"),
                model_type="text-generation",
                quantization=quant_level,
                device="auto",
                validate=True
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            test_result = {
                "success": result["success"],
                "duration": duration,
                "error": result.get("error"),
                "model_validation": result.get("model_validation", {}),
                "output_file": str(self.output_dir / f"gguf_gpu_{quant_level}.gguf")
            }
            
            if result["success"]:
                output_file = Path(test_result["output_file"])
                if output_file.exists():
                    test_result["file_size_mb"] = output_file.stat().st_size / 1024**2
                    logger.info(f"GGUF量化 {quant_level} 成功，耗时: {duration:.2f}秒，文件大小: {test_result['file_size_mb']:.2f}MB")
                else:
                    logger.error(f"GGUF量化 {quant_level} 成功但输出文件不存在")
            else:
                logger.error(f"GGUF量化 {quant_level} 失败: {result.get('error')}")
                
            results[quant_level] = test_result
            
        return results
    
    def test_cpu_vs_gpu_performance(self) -> Dict[str, Any]:
        """对比CPU和GPU性能"""
        logger.info("=== CPU vs GPU性能对比测试 ===")
        
        # 强制使用CPU
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["MPS_VISIBLE_DEVICES"] = ""
        
        start_time = time.time()
        cpu_result = self.converter.convert(
            input_source=self.test_model,
            output_format="gptq",
            output_path=str(self.output_dir / "gptq_cpu"),
            model_type="text-generation",
            device="cpu",
            validate=False
        )
        cpu_duration = time.time() - start_time
        
        # 恢复GPU使用
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ.pop("MPS_VISIBLE_DEVICES", None)
        
        start_time = time.time()
        gpu_result = self.converter.convert(
            input_source=self.test_model,
            output_format="gptq",
            output_path=str(self.output_dir / "gptq_gpu_compare"),
            model_type="text-generation",
            device="auto",
            validate=False
        )
        gpu_duration = time.time() - start_time
        
        performance_comparison = {
            "cpu": {
                "success": cpu_result["success"],
                "duration": cpu_duration,
                "error": cpu_result.get("error")
            },
            "gpu": {
                "success": gpu_result["success"],
                "duration": gpu_duration,
                "error": gpu_result.get("error")
            }
        }
        
        if cpu_result["success"] and gpu_result["success"]:
            speedup = cpu_duration / gpu_duration
            logger.info(f"CPU耗时: {cpu_duration:.2f}秒")
            logger.info(f"GPU耗时: {gpu_duration:.2f}秒")
            logger.info(f"GPU加速比: {speedup:.2f}x")
            performance_comparison["speedup"] = speedup
        else:
            logger.error("性能对比测试失败")
            
        return performance_comparison
    
    def test_model_loading_speed(self) -> Dict[str, Any]:
        """测试模型加载速度"""
        logger.info("=== 模型加载速度测试 ===")
        
        # 测试原始模型加载
        start_time = time.time()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model = AutoModelForCausalLM.from_pretrained(self.test_model)
            tokenizer = AutoTokenizer.from_pretrained(self.test_model)
            original_load_time = time.time() - start_time
            logger.info(f"原始模型加载时间: {original_load_time:.2f}秒")
        except Exception as e:
            logger.error(f"原始模型加载失败: {e}")
            original_load_time = None
        
        # 测试量化模型加载
        quantized_path = self.output_dir / "gptq_gpu"
        if quantized_path.exists():
            start_time = time.time()
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                model = AutoModelForCausalLM.from_pretrained(str(quantized_path))
                tokenizer = AutoTokenizer.from_pretrained(str(quantized_path))
                quantized_load_time = time.time() - start_time
                logger.info(f"量化模型加载时间: {quantized_load_time:.2f}秒")
            except Exception as e:
                logger.error(f"量化模型加载失败: {e}")
                quantized_load_time = None
        else:
            quantized_load_time = None
            
        return {
            "original_load_time": original_load_time,
            "quantized_load_time": quantized_load_time
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        logger.info("开始GPU量化转换测试")
        
        test_results = {
            "gpu_info": self.check_gpu_availability(),
            "device_detection": self.test_device_detection(),
            "gptq_quantization": self.test_gptq_quantization_gpu(),
            "awq_quantization": self.test_awq_quantization_gpu(),
            "gguf_quantization": self.test_gguf_quantization_gpu(),
            "performance_comparison": self.test_cpu_vs_gpu_performance(),
            "model_loading_speed": self.test_model_loading_speed()
        }
        
        # 生成测试报告
        self.generate_test_report(test_results)
        
        return test_results
    
    def generate_test_report(self, results: Dict[str, Any]):
        """生成测试报告"""
        logger.info("=== GPU量化转换测试报告 ===")
        
        # GPU信息
        gpu_info = results["gpu_info"]
        logger.info(f"GPU状态: {'CUDA' if gpu_info['cuda_available'] else 'MPS' if gpu_info['mps_available'] else 'CPU'}")
        
        # 量化测试结果
        gptq_result = results["gptq_quantization"]
        awq_result = results["awq_quantization"]
        gguf_results = results["gguf_quantization"]
        
        logger.info(f"GPTQ量化: {'成功' if gptq_result['success'] else '失败'}")
        if gptq_result['success']:
            logger.info(f"  - 耗时: {gptq_result['duration']:.2f}秒")
            
        logger.info(f"AWQ量化: {'成功' if awq_result['success'] else '失败'}")
        if awq_result['success']:
            logger.info(f"  - 耗时: {awq_result['duration']:.2f}秒")
            
        logger.info("GGUF量化结果:")
        for level, result in gguf_results.items():
            status = "成功" if result['success'] else "失败"
            logger.info(f"  - {level}: {status}")
            if result['success']:
                logger.info(f"    - 耗时: {result['duration']:.2f}秒")
                if 'file_size_mb' in result:
                    logger.info(f"    - 文件大小: {result['file_size_mb']:.2f}MB")
        
        # 性能对比
        perf_comparison = results["performance_comparison"]
        if "speedup" in perf_comparison:
            logger.info(f"GPU加速比: {perf_comparison['speedup']:.2f}x")
        
        # 模型加载速度
        loading_speed = results["model_loading_speed"]
        if loading_speed["original_load_time"] and loading_speed["quantized_load_time"]:
            load_speedup = loading_speed["original_load_time"] / loading_speed["quantized_load_time"]
            logger.info(f"量化模型加载加速比: {load_speedup:.2f}x")

def main():
    """主函数"""
    tester = GPUQuantizationTester()
    results = tester.run_all_tests()
    
    # 保存测试结果到文件
    import json
    with open("gpu_quantization_test_results.json", "w", encoding="utf-8") as f:
        # 移除不可序列化的对象
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, (str, int, float, bool, list)):
                        serializable_results[key][k] = v
            elif isinstance(value, (str, int, float, bool)):
                serializable_results[key] = value
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    logger.info("测试完成，结果已保存到 gpu_quantization_test_results.json")

if __name__ == "__main__":
    main() 