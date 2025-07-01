"""
Model Converter Tool - A comprehensive AI model format conversion utility

This package provides tools for converting AI models between different formats
including HuggingFace, ONNX, GGUF, MLX, TorchScript, GPTQ, AWQ, and more.

Main components:
- ModelConverter: Core conversion engine
- ModelValidator: Model validation and testing
- ConfigManager: Configuration and preset management
- CLI: Command-line interface
- Utils: Utility functions

Example usage:
    from model_converter_tool import ModelConverter

    converter = ModelConverter()
    result = converter.convert(
        input_source="gpt2",
        output_format="onnx",
        output_path="./output/gpt2.onnx"
    )
"""

import os
# 在模块导入前就禁用 MPS/CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MPS_VISIBLE_DEVICES"] = ""
os.environ["TRANSFORMERS_NO_MPS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["USE_CPU_ONLY"] = "1"

# 强制设置 PyTorch 相关环境变量
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.0"

# 禁用所有可能的 GPU 后端
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:0"

# Import main classes and functions
from .converter import ModelConverter
from .validator import ModelValidator
from .config import (
    ConfigManager,
    ConversionConfig,
    load_config_preset,
    list_available_presets,
)
from .cli import (
    cli,
    detect_model_format,
    load_model_with_fallbacks,
    validate_conversion_compatibility,
)
from .utils import (
    setup_directories,
    cleanup_temp_files,
    get_file_size,
    format_file_size,
    get_directory_size,
    create_temp_directory,
    safe_filename,
    is_valid_model_path,
    get_model_name_from_path,
    ensure_output_directory,
    copy_with_progress,
    remove_directory_safely,
    create_dummy_model,
)

# Version information
__version__ = "1.0.0"
__author__ = "Model Converter Tool Team"
__email__ = "support@modelconverter.com"
__description__ = "A comprehensive AI model format conversion utility"
__url__ = "https://github.com/modelconverter/model-converter-tool"

# Define what gets imported with "from model_converter_tool import *"
__all__ = [
    # Main classes
    "ModelConverter",
    "ModelValidator",
    "ConfigManager",
    "ConversionConfig",
    # CLI functions
    "cli",
    "detect_model_format",
    "load_model_with_fallbacks",
    "validate_conversion_compatibility",
    # Configuration functions
    "load_config_preset",
    "list_available_presets",
    # Utility functions
    "setup_directories",
    "cleanup_temp_files",
    "get_file_size",
    "format_file_size",
    "get_directory_size",
    "create_temp_directory",
    "safe_filename",
    "is_valid_model_path",
    "get_model_name_from_path",
    "ensure_output_directory",
    "copy_with_progress",
    "remove_directory_safely",
    "create_dummy_model",
]


def quick_convert(input_source: str, output_format: str, output_path: str, **kwargs):
    """
    Quick conversion function for simple use cases.

    Args:
        input_source: Path to input model or HuggingFace model name
        output_format: Target format (hf, onnx, gguf, mlx, torchscript, fp16, gptq, awq)
        output_path: Output path for converted model
        **kwargs: Additional conversion parameters

    Returns:
        Dict with conversion results
    """
    converter = ModelConverter()
    return converter.convert(
        input_source=input_source,
        output_format=output_format,
        output_path=output_path,
        **kwargs,
    )


# Add quick_convert to __all__
__all__.append("quick_convert")


def batch_convert(tasks: list, **kwargs):
    """
    Batch conversion function for multiple models.

    Args:
        tasks: List of conversion task dictionaries
        **kwargs: Additional batch conversion parameters

    Returns:
        List of conversion results
    """
    converter = ModelConverter()
    return converter.batch_convert(tasks, **kwargs)


# Add batch_convert to __all__
__all__.append("batch_convert")


def validate_model(model_path: str, output_format: str, model_type: str = "text-generation", **kwargs):
    """
    Quick validation function for converted models.

    Args:
        model_path: Path to the model to validate
        output_format: Format of the model
        model_type: Type of model
        **kwargs: Additional validation parameters

    Returns:
        Dict with validation results
    """
    validator = ModelValidator()
    return validator.validate_converted_model(
        model_path=model_path,
        output_format=output_format,
        model_type=model_type,
        **kwargs,
    )


# Add validate_model to __all__
__all__.append("validate_model")


def get_supported_formats():
    """
    Get all supported input and output formats.

    Returns:
        Dict with supported formats
    """
    converter = ModelConverter()
    return converter.get_supported_formats()


# Add get_supported_formats to __all__
__all__.append("get_supported_formats")


def _initialize_package():
    """Initialize package on import"""
    import logging

    # Set up basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create necessary directories
    try:
        setup_directories()
    except Exception:
        pass  # Silently fail if directories can't be created


# Run initialization
_initialize_package()
