import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Union, Tuple

logger = logging.getLogger(__name__)

# Supported format list
SUPPORTED_FORMATS = ["onnx", "torchscript", "gguf", "fp16", "awq", "gptq", "hf", "safetensors", "mlx"]

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
        elif output_format == "fp16":
            from .engine.fp16 import convert_to_fp16, validate_fp16_file
            return convert_to_fp16, validate_fp16_file
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
        """
        Validate conversion inputs and return validation result.
        
        Args:
            input_format: Input model format
            output_format: Output model format
            model_type: Model type
            quantization: Quantization parameters
            device: Target device
            
        Returns:
            Validation result dictionary
        """
        errors = []
        warnings = []
        
        # Check if input format is supported
        if input_format not in SUPPORTED_FORMATS and input_format != "huggingface":
            errors.append(f"Unsupported input format: {input_format}")
        
        # Check if output format is supported
        if output_format not in SUPPORTED_FORMATS:
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
        if quantization and output_format not in ["gguf", "gptq", "awq", "mlx"]:
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
    ) -> ConversionResult:
        """
        Convert a model to the specified format.
        Args:
            model: Loaded model object (optional)
            tokenizer: Loaded tokenizer object (optional)
            model_name: Source model name or path (optional, auto-load if model/tokenizer not provided)
            output_format: Target format
            output_path: Output path
            model_type: Model type
            device: Device
            quantization: Quantization parameters (optional)
            use_large_calibration: Whether to use large-scale calibration
        Returns:
            ConversionResult dataclass
        """
        result = ConversionResult(success=False)
        try:
            # Auto-load transformers model and tokenizer
            if (model is None or tokenizer is None) and model_name is not None:
                from model_converter_tool.utils import load_model_with_cache, load_tokenizer_with_cache
                model = load_model_with_cache(model_name)
                tokenizer = load_tokenizer_with_cache(model_name)
            if output_format in SUPPORTED_FORMATS:
                convert_func, validate_func = self._get_converter_functions(output_format)
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