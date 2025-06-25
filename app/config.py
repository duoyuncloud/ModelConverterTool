"""
config.py
Global configuration and environment loading with improved settings.
"""

import os
from typing import Dict, Any, List
from pydantic_settings import BaseSettings
import logging

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """Application configuration with improved defaults and validation."""
    app_name: str = "Model Converter Tool"
    version: str = "3.0.0"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    upload_dir: str = "./uploads"
    output_dir: str = "./outputs"
    redis_url: str = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    max_file_size: int = 1024 * 1024 * 1024
    cors_origins: List[str] = ["*"]
    supported_formats: List[str] = ["onnx", "torchscript", "fp16", "hf", "gptq", "awq", "gguf", "mlx", "test"]
    supported_model_types: List[str] = ["causal_lm", "seq2seq", "encoder"]
    cache_ttl: int = 3600 * 24
    max_cache_size: int = 10
    task_timeout_minutes: int = 10
    max_concurrent_tasks: int = 3
    monitoring_interval: int = 30
    log_level: str = "INFO"
    model_cache_enabled: bool = True
    max_model_cache_size: int = 5
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()

def validate_config() -> bool:
    """Validate configuration and ensure directories and Redis are available."""
    try:
        os.makedirs(settings.upload_dir, exist_ok=True)
        os.makedirs(settings.output_dir, exist_ok=True)
        try:
            import redis
            redis_client = redis.from_url(settings.redis_url)
            redis_client.ping()
            logger.info(f"Redis connection successful: {settings.redis_url}")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            logger.warning("The application will continue but background tasks may not work.")
        if not settings.supported_formats:
            raise ValueError("No supported formats configured")
        if not settings.supported_model_types:
            raise ValueError("No supported model types configured")
        logger.info(f"Configuration validated successfully")
        logger.info(f"Supported formats: {', '.join(settings.supported_formats)}")
        logger.info(f"Supported model types: {', '.join(settings.supported_model_types)}")
        return True
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False

def get_format_config(format_name: str) -> Dict[str, Any]:
    """Get configuration for a specific format."""
    format_configs = {
        "onnx": {
            "description": "Open Neural Network Exchange format",
            "extensions": [".onnx"],
            "requires_gpu": False
        },
        "torchscript": {
            "description": "PyTorch TorchScript format",
            "extensions": [".pt", ".pth"],
            "requires_gpu": False
        },
        "fp16": {
            "description": "Half-precision floating point format",
            "extensions": [""],
            "requires_gpu": False
        },
        "hf": {
            "description": "HuggingFace format",
            "extensions": [""],
            "requires_gpu": False
        },
        "gptq": {
            "description": "GPTQ quantization format",
            "extensions": [""],
            "requires_gpu": True
        },
        "awq": {
            "description": "AWQ quantization format",
            "extensions": [""],
            "requires_gpu": True
        },
        "gguf": {
            "description": "GGUF format for llama.cpp",
            "extensions": [".gguf"],
            "requires_gpu": False
        },
        "mlx": {
            "description": "Apple MLX format",
            "extensions": [".npz"],
            "requires_gpu": False
        },
        "test": {
            "description": "Test format for validation",
            "extensions": [".json"],
            "requires_gpu": False
        }
    }
    return format_configs.get(format_name.lower(), {})

def get_system_info() -> Dict[str, Any]:
    """Get system information for monitoring."""
    import platform
    import psutil
    return {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_total": psutil.virtual_memory().total,
        "memory_available": psutil.virtual_memory().available,
        "disk_usage": psutil.disk_usage('/').percent
    } 