"""
Utility functions for model conversion
"""

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Optional, Dict, List, Union, Tuple

import torch

logger = logging.getLogger(__name__)


def setup_directories() -> None:
    """Setup required directories for the tool"""
    directories = ["outputs", "uploads", "model_cache", "configs", "temp"]

    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.debug(f"Created directory: {directory}")


def cleanup_temp_files() -> None:
    """Clean up temporary files"""
    temp_dir = Path("temp")
    if temp_dir.exists():
        try:
            shutil.rmtree(temp_dir)
            temp_dir.mkdir(exist_ok=True)
            logger.debug("Cleaned up temporary files")
        except Exception as e:
            logger.warning(f"Could not clean up temp files: {e}")


def get_file_size(file_path: str) -> Optional[int]:
    """Get file size in bytes"""
    try:
        return Path(file_path).stat().st_size
    except Exception:
        return None


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes = size_bytes / 1024.0
        i += 1

    return f"{size_bytes:.1f}{size_names[i]}"


def get_directory_size(directory: str) -> int:
    """Get total size of directory in bytes"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
    except Exception as e:
        logger.warning(f"Could not calculate directory size: {e}")

    return total_size


def create_temp_directory() -> str:
    """Create a temporary directory and return its path"""
    temp_dir = Path("temp") / f"conv_{os.getpid()}_{int(time.time())}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return str(temp_dir)


def safe_filename(filename: str) -> str:
    """Convert filename to safe version"""
    # Remove or replace unsafe characters
    unsafe_chars = '<>:"/\\|?*'
    for char in unsafe_chars:
        filename = filename.replace(char, "_")

    # Limit length
    if len(filename) > 200:
        name, ext = os.path.splitext(filename)
        filename = name[: 200 - len(ext)] + ext

    return filename


def is_valid_model_path(path: str) -> bool:
    """Check if path is a valid model path"""
    if not path:
        return False

    # Check if it's a HuggingFace model name
    if path.startswith("hf:"):
        return True

    # Check if it's a valid local path
    try:
        return Path(path).exists()
    except Exception:
        return False


def get_model_name_from_path(path: str) -> str:
    """Extract model name from path"""
    if path.startswith("hf:"):
        return path[3:].split("/")[-1]
    else:
        return Path(path).name


def ensure_output_directory(output_path: str) -> str:
    """Ensure output directory exists and return the path"""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_path)


def copy_with_progress(src: str, dst: str, description: str = "Copying") -> bool:
    """Copy file with progress indication"""
    try:
        src_path = Path(src)
        dst_path = Path(dst)

        if not src_path.exists():
            logger.error(f"Source file does not exist: {src}")
            return False

        # Create destination directory if needed
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy file
        shutil.copy2(src_path, dst_path)
        logger.info(f"✅ {description}: {src} -> {dst}")
        return True

    except Exception as e:
        logger.error(f"❌ {description} failed: {e}")
        return False


def remove_directory_safely(directory: str) -> bool:
    """Safely remove a directory"""
    try:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            logger.debug(f"Removed directory: {directory}")
        return True
    except Exception as e:
        logger.warning(f"Could not remove directory {directory}: {e}")
        return False


def create_dummy_model(output_dir: str, **kwargs):
    """
    生成简单的测试用 dummy model。
    注意：此函数仅用于测试，生产环境请使用真实模型。
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 简化的 config.json
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 128,
        "intermediate_size": 512,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "vocab_size": 1000,
        "max_position_embeddings": 512,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "torch_dtype": "float16",
        **kwargs,
    }
    
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    
    # 简化的 tokenizer 文件
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "model_max_length": 512,
    }
    with open(os.path.join(output_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, indent=2)
    
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
    for i in range(4, 100):
        vocab[f"token{i}"] = i
    
    with open(os.path.join(output_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)
    
    logger.info(f"✅ Simple dummy model config generated at {output_dir}")
    logger.warning("⚠️  This is a test-only dummy model. Use real models in production.")
