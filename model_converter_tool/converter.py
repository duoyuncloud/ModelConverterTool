import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Union, Tuple

logger = logging.getLogger(__name__)

# Supported format list
SUPPORTED_FORMATS = ["onnx", "torchscript", "gguf", "awq", "gptq", "hf", "safetensors", "mlx"]

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
    API-First: All parameters explicit, type-safe, well-documented, returns dataclass.
    """
    
    def _get_converter_functions(self, output_format: str):
        """Lazy import converter functions"""
        if output_format == "onnx":
            from .engine.onnx import convert_to_onnx, validate_onnx_file
            return convert_to_onnx, validate_onnx_file
        elif output_format == "torchscript":
            from .engine.torchscript import convert_to_torchscript, validate_torchscript_file
            return convert_to_torchscript, validate_torchscript_file
        elif output_format == "gguf":
            from .engine.gguf import convert_to_gguf, validate_gguf_file
            return convert_to_gguf, validate_gguf_file
        elif output_format == "awq":
            from .engine.awq import convert_to_awq, validate_awq_file
            return convert_to_awq, validate_awq_file
        elif output_format == "gptq":
            from .engine.gptq import convert_to_gptq, validate_gptq_file
            return convert_to_gptq, validate_gptq_file
        elif output_format == "hf":
            from .engine.hf import convert_to_hf, validate_hf_file
            return convert_to_hf, validate_hf_file
        elif output_format == "safetensors":
            from .engine.safetensors import convert_to_safetensors, validate_safetensors_file
            return convert_to_safetensors, validate_safetensors_file
        elif output_format == "mlx":
            from .engine.mlx import convert_to_mlx, validate_mlx_file
            return convert_to_mlx, validate_mlx_file
        elif output_format == "custom_quant":
            from .engine.custom_quant import convert_to_custom_quant, validate_custom_quant_file
            return convert_to_custom_quant, validate_custom_quant_file
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _detect_model_format(self, input_model: str) -> Tuple[str, str]:
        """
        Detect the format of input model.
        
        Args:
            input_model: Model path or name
            
        Returns:
            Tuple of (format, normalized_path)
        """
        path = Path(input_model)
        
        # First check file extension (even if file doesn't exist)
        suffix = path.suffix.lower()
        if suffix == ".onnx":
            return "onnx", str(path)
        elif suffix == ".gguf":
            return "gguf", str(path)
        elif suffix in [".pt", ".pth"]:
            return "torchscript", str(path)
        elif suffix == ".safetensors":
            return "safetensors", str(path)
        elif suffix == ".npz":
            return "mlx", str(path)
        
        # If local doesn't exist and no clear file extension, consider it Hugging Face Hub name
        if not path.exists():
            # Check if it contains slashes (might be organization/model name format) or model name without extension
            if "/" in input_model or "\\" in input_model or (not suffix and not input_model.startswith(".")):
                return "huggingface", input_model
            else:
                return "unknown", str(path)
        
        # Check if it's a local file
        if path.is_file():
            # File exists but no supported extension
            return "unknown", str(path)
        
        # Check if it's a directory (likely Hugging Face format)
        if path.is_dir():
            # Check for Hugging Face format indicators
            if (path / "config.json").exists() or (path / "pytorch_model.bin").exists() or any(path.glob("*.safetensors")):
                return "huggingface", str(path)
            else:
                return "unknown", str(path)
        
        # Default to unknown
        return "unknown", str(path)
    
    def _validate_conversion_inputs(
        self, 
        input_format: str, 
        output_format: str, 
        model_type: str, 
        quantization: str = "", 
        device: str = "auto"
    ) -> Dict[str, Any]:
        errors = []
        warnings = []

        # Check if input format is supported
        if input_format not in SUPPORTED_FORMATS and input_format != "huggingface":
            errors.append(f"Unsupported input format: {input_format}")

        # Check if output format is supported (allow custom_quant for all)
        valid_outputs = self.get_conversion_matrix().get(input_format, [])
        if output_format not in valid_outputs:
            errors.append(f"Unsupported output format: {output_format}")

        # Check model type
        valid_model_types = ["auto", "text-generation", "text-classification", "image-classification"]
        if model_type not in valid_model_types:
            errors.append(f"Invalid model type: {model_type}. Valid types: {valid_model_types}")

        # Check device
        valid_devices = ["auto", "cpu", "cuda", "mps"]
        if device not in valid_devices:
            errors.append(f"Invalid device: {device}. Valid devices: {valid_devices}")

        # Check quantization for supported formats
        if quantization and output_format not in ["gguf", "gptq", "awq", "mlx", "custom_quant"]:
            warnings.append(f"Quantization not supported for {output_format} format")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }

    def convert(
        self,
        model: Any = None,
        tokenizer: Any = None,
        model_name: str = None,
        output_format: str = None,
        output_path: str = None,
        model_type: str = "auto",
        device: str = "auto",
        quantization: Optional[str] = None,
        use_large_calibration: bool = False,
        dtype: str = None,
        quantization_config: dict = None,
    ) -> ConversionResult:
        result = ConversionResult(success=False)
        try:
            input_format, norm_path = self._detect_model_format(model_name)
            # Special dispatch for safetensors
            if output_format == "safetensors":
                try:
                    import torch
                    from model_converter_tool.engine.safetensors import convert_to_safetensors, validate_safetensors_file
                    if model is None:
                        from transformers import AutoModel
                        from model_converter_tool.utils import load_model_with_cache
                        model = load_model_with_cache(norm_path, AutoModel)
                    success, extra_info = convert_to_safetensors(
                        model,
                        tokenizer,
                        model_name,
                        output_path,
                        model_type,
                        device,
                        dtype
                    )
                    result.success = success
                    result.extra_info = extra_info
                    result.output_path = output_path
                    return result
                except Exception as e:
                    result.error = f"Safetensors conversion failed: {e}"
                    return result
            # Special dispatch for custom_quant
            if output_format == "custom_quant":
                try:
                    import torch
                    from model_converter_tool.engine.custom_quant import convert_to_custom_quant, validate_custom_quant_file
                    if model is None:
                        from transformers import AutoModel
                        from model_converter_tool.utils import load_model_with_cache
                        model = load_model_with_cache(norm_path, AutoModel)
                    # tokenizer is not used in custom_quant
                    success, extra_info = convert_to_custom_quant(
                        model,
                        None,
                        model_name,
                        output_path,
                        model_type,
                        device,
                        quantization,
                        use_large_calibration,
                        quantization_config
                    )
                    result.success = success
                    result.extra_info = extra_info
                    result.output_path = output_path
                    return result
                except Exception as e:
                    result.error = f"Custom quantization failed: {e}"
                    return result
            # For ONNX, only pass string arguments, do not pass model/tokenizer objects
            if output_format == "onnx":
                convert_func, validate_func = self._get_converter_functions(output_format)
                success = convert_func(
                    model_name=model_name,
                    output_path=output_path,
                    device=device
                )
                result.success = success
                result.validation = success
                result.output_path = output_path
                result.extra_info = None
                if not success:
                    result.error = f"ONNX conversion or validation failed for {model_name}"
            elif output_format in SUPPORTED_FORMATS:
                convert_func, validate_func = self._get_converter_functions(output_format)
                if output_format == "safetensors":
                    success, extra = convert_func(
                        model, tokenizer, model_name, output_path, model_type, device, dtype
                    )
                    if not success and isinstance(extra, str):
                        result.error = extra
                elif output_format in ("awq", "gptq"):
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
                    if isinstance(extra, str):
                        result.error = extra
                    else:
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
        Batch convert multiple models.
        Args:
            tasks: Task list, each task is a dict with same parameters as convert
            max_workers: Number of concurrent workers
            max_retries: Maximum retry attempts
        Returns:
            List of ConversionResult
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

    def get_conversion_matrix(self) -> Dict[str, List[str]]:
        """Public method to get the conversion matrix, including custom_quant support."""
        matrix = self._get_conversion_matrix()
        # Add 'custom_quant' to all input types
        for k in matrix:
            if 'custom_quant' not in matrix[k]:
                matrix[k].append('custom_quant')
        return matrix

    def _get_conversion_matrix(self) -> Dict[str, List[str]]:
        return {
            "huggingface": ["huggingface", "safetensors", "torchscript", "onnx", "gguf", "mlx"],
            "safetensors": ["huggingface", "safetensors"],
            "torchscript": ["torchscript"],
            "onnx": ["onnx"],
            "gguf": ["gguf"],
            "mlx": ["mlx"],
        } 