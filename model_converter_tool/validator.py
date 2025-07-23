# Advanced inference validation and quantization quality evaluation tool

import logging
from typing import Any, Dict, List
from model_converter_tool.utils import load_model_and_tokenizer
import torch

logger = logging.getLogger(__name__)


class ModelValidator:
    """
    Simplified inference validation and quantized model quality evaluation
    """

    def __init__(self, original_model_path: str, quantized_model_path: str):
        self.original_model_path = original_model_path
        self.quantized_model_path = quantized_model_path

    def compare_models(self, test_cases: List[str]) -> Dict[str, Any]:
        results = {}
        try:
            original_model, original_tokenizer = load_model_and_tokenizer(self.original_model_path)
            quantized_model, quantized_tokenizer = load_model_and_tokenizer(self.quantized_model_path)
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return {"error": f"Model loading failed: {e}"}
        for i, test_case in enumerate(test_cases):
            try:
                orig_out = self._inference(original_model, original_tokenizer, test_case)
                quant_out = self._inference(quantized_model, quantized_tokenizer, test_case)
                results[f"test_case_{i}"] = {
                    "input": test_case,
                    "original_output": orig_out,
                    "quantized_output": quant_out,
                    "similarity": self._calculate_similarity(orig_out, quant_out),
                }
            except Exception as e:
                logger.error(f"Test case {i} failed: {e}")
                results[f"test_case_{i}"] = {"error": str(e)}
        return results

    def _inference(self, model, tokenizer, text: str) -> str:
        try:
            inputs = tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            return tokenizer.decode(outputs.logits.argmax(dim=-1)[0])
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return ""

    def _calculate_similarity(self, output1: str, output2: str) -> float:
        if not output1 or not output2:
            return 0.0
        return sum(1 for a, b in zip(output1, output2) if a == b) / max(len(output1), len(output2))
