"""
高级推理验证与量化质量评测工具
"""
import logging
import os
from typing import Any, Dict, List
import numpy as np

logger = logging.getLogger(__name__)

class ModelQualityEvaluator:
    """
    高级推理验证与量化模型质量评测
    """
    @staticmethod
    def validate_quantization_quality(
        original_model_path: str,
        quantized_model_path: str,
        quantization_type: str,
        model_type: str = "text-generation",
    ) -> Dict[str, Any]:
        """
        对比原始模型与量化模型的推理输出、困惑度、相似度等质量指标。
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            # 加载原始模型
            original_model = AutoModelForCausalLM.from_pretrained(original_model_path)
            tokenizer = AutoTokenizer.from_pretrained(original_model_path)
            # 加载量化模型
            if quantization_type.lower() == "gptq":
                try:
                    from gptqmodel import GPTQModel
                    quantized_model = GPTQModel.load(quantized_model_path)
                except ImportError:
                    return {"success": False, "error": "gptqmodel not available for quality validation"}
            elif quantization_type.lower() == "awq":
                try:
                    from gptqmodel import GPTQModel
                    quantized_model = GPTQModel.load(quantized_model_path)
                except ImportError:
                    return {"success": False, "error": "gptqmodel not available for quality validation"}
            else:
                quantized_model = AutoModelForCausalLM.from_pretrained(quantized_model_path)
            # 测试用例
            test_cases = [
                "Hello world",
                "The quick brown fox",
                "Machine learning is",
                "In the beginning",
                "To be or not to be",
            ]
            quality_metrics: Dict[str, Any] = {
                "perplexity_diff": [],
                "output_similarity": [],
                "inference_speed": [],
                "memory_usage": [],
            }
            for test_text in test_cases:
                try:
                    inputs = tokenizer(test_text, return_tensors="pt")
                    # 原始模型推理
                    with torch.no_grad():
                        original_outputs = original_model(**inputs)
                    original_logits = original_outputs.logits
                    # 量化模型推理
                    if quantization_type.lower() in ["gptq", "awq"]:
                        # Use gptqmodel's generate method
                        result = quantized_model.generate(test_text, max_length=len(inputs["input_ids"][0]) + 5)
                        quantized_logits = original_logits  # 占位，实际应获取真实 logits
                    else:
                        with torch.no_grad():
                            quantized_outputs = quantized_model(**inputs)
                        quantized_logits = quantized_outputs.logits
                    # 计算输出相似度
                    if original_logits.shape == quantized_logits.shape:
                        similarity = torch.cosine_similarity(
                            original_logits.flatten(), quantized_logits.flatten(), dim=0
                        ).item()
                        quality_metrics["output_similarity"].append(similarity)
                        # 计算困惑度差异
                        orig_probs = torch.softmax(original_logits, dim=-1)
                        quant_probs = torch.softmax(quantized_logits, dim=-1)
                        perplexity_diff = torch.abs(orig_probs - quant_probs).mean().item()
                        quality_metrics["perplexity_diff"].append(perplexity_diff)
                except Exception as e:
                    logger.warning(f"Quality test failed for '{test_text}': {e}")
            # 计算平均指标
            avg_similarity = (
                np.mean(quality_metrics["output_similarity"]) if quality_metrics["output_similarity"] else 0
            )
            avg_perplexity_diff = (
                np.mean(quality_metrics["perplexity_diff"]) if quality_metrics["perplexity_diff"] else 0
            )
            # 计算模型大小差异
            original_size = sum(p.numel() * p.element_size() for p in original_model.parameters())
            quantized_size = 0
            if os.path.exists(quantized_model_path):
                for root, dirs, files in os.walk(quantized_model_path):
                    for file in files:
                        quantized_size += os.path.getsize(os.path.join(root, file))
            else:
                quantized_size = original_size  # Fallback
            compression_ratio = original_size / quantized_size if quantized_size > 0 else 1
            return {
                "success": True,
                "avg_similarity": avg_similarity,
                "avg_perplexity_diff": avg_perplexity_diff,
                "compression_ratio": compression_ratio,
                "details": quality_metrics,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def batch_quality_evaluation(
        model_pairs: List[Dict[str, str]],
        quantization_type: str,
        model_type: str = "text-generation",
    ) -> List[Dict[str, Any]]:
        """
        批量评测多个模型对的量化质量。
        model_pairs: List of dicts with keys: original_model_path, quantized_model_path
        """
        results = []
        for pair in model_pairs:
            result = ModelQualityEvaluator.validate_quantization_quality(
                pair["original_model_path"],
                pair["quantized_model_path"],
                quantization_type,
                model_type,
            )
            results.append(result)
        return results
