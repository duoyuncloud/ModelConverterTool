import logging
from pathlib import Path
from .engine.onnx import convert_to_onnx, validate_onnx_file
from .engine.torchscript import convert_to_torchscript, validate_torchscript_file
from .engine.gguf import convert_to_gguf, validate_gguf_file
from .engine.fp16 import convert_to_fp16, validate_fp16_file
from .engine.awq import convert_to_awq, validate_awq_file
from .engine.gptq import convert_to_gptq, validate_gptq_file
from .engine.hf import convert_to_hf, validate_hf_file
from .engine.safetensors import convert_to_safetensors, validate_safetensors_file
from .engine.mlx import convert_to_mlx, validate_mlx_file
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Union, Tuple

logger = logging.getLogger(__name__)

CONVERTERS = {
    "onnx": {"convert": convert_to_onnx, "validate": validate_onnx_file},
    "torchscript": {"convert": convert_to_torchscript, "validate": validate_torchscript_file},
    "gguf": {"convert": convert_to_gguf, "validate": validate_gguf_file},
    "fp16": {"convert": convert_to_fp16, "validate": validate_fp16_file},
    "awq": {"convert": convert_to_awq, "validate": validate_awq_file},
    "gptq": {"convert": convert_to_gptq, "validate": validate_gptq_file},
    "hf": {"convert": convert_to_hf, "validate": validate_hf_file},
    "safetensors": {"convert": convert_to_safetensors, "validate": validate_safetensors_file},
    "mlx": {"convert": convert_to_mlx, "validate": validate_mlx_file},
}

@dataclass
class ConversionResult:
    success: bool
    error: Optional[str] = None
    output_path: Optional[str] = None
    validation: Optional[bool] = None
    extra_info: Optional[Dict[str, Any]] = None

class ModelConverter:
    """
    Model converter with engine-based logic and unified dispatch.
    API-First: 参数全部显式、类型安全、文档齐全，返回 dataclass。
    """
    def convert(
        self,
        model: Any,
        tokenizer: Any,
        model_name: str,
        output_format: str,
        output_path: str,
        model_type: str = "auto",
        device: str = "auto",
        quantization: Optional[str] = None,
        use_large_calibration: bool = False,
    ) -> ConversionResult:
        """
        Convert a model to the specified format.
        Args:
            model: 已加载的模型对象
            tokenizer: 已加载的分词器对象
            model_name: 源模型名称或路径
            output_format: 目标格式
            output_path: 输出路径
            model_type: 模型类型
            device: 设备
            quantization: 量化参数（可选）
            use_large_calibration: 是否使用大范围校准
        Returns:
            ConversionResult dataclass
        """
        result = ConversionResult(success=False)
        try:
            if output_format in CONVERTERS:
                convert_func = CONVERTERS[output_format]["convert"]
                validate_func = CONVERTERS[output_format]["validate"]
                if output_format in ("awq", "gptq"):
                    success, extra = convert_func(
                        model, tokenizer, model_name, output_path, model_type, device, quantization, use_large_calibration
                    )
                else:
                    success, extra = convert_func(
                        model, tokenizer, model_name, output_path, model_type, device
                    )
                if success:
                    valid = validate_func(Path(output_path), extra)
                    result.success = valid
                    result.validation = valid
                    result.output_path = output_path
                    result.extra_info = extra
                else:
                    result.success = False
                    result.error = f"Conversion failed for {output_format}"
            else:
                result.error = f"Unsupported format: {output_format}"
        except Exception as e:
            result.success = False
            result.error = str(e)
        return result

    def batch_convert(
        self,
        tasks: List[Dict[str, Any]],
        max_workers: int = 1,
        max_retries: int = 1,
    ) -> List[ConversionResult]:
        """
        Batch convert models with unified API.
        Args:
            tasks: 任务列表，每个任务为 dict，参数同 convert
            max_workers: 并发数
            max_retries: 最大重试次数
        Returns:
            List[ConversionResult]
        """
        results: List[ConversionResult] = []
        for task in tasks:
            retries = 0
            while retries < max_retries:
                try:
                    res = self.convert(**task)
                    results.append(res)
                    break
                except Exception as e:
                    retries += 1
                    if retries >= max_retries:
                        results.append(ConversionResult(success=False, error=str(e)))
        return results 