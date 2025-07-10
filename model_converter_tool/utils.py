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
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

logger = logging.getLogger(__name__)
console = Console()


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
    Generate a simple test dummy model.
    Note: This function is for testing only, please use real models in production.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Simplified config.json
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
    
    # Simplified tokenizer files
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


def get_local_cache_path(model_name: str) -> str:
    """
    Get the path of the model in local cache
    
    Args:
        model_name: Model name (e.g., "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    Returns:
        Local cache path, if not exists return original model name
    """
    import os
    from pathlib import Path
    
    # Build cache directory path
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_cache_dir = cache_dir / f"models--{model_name.replace('/', '--')}"
    
    if not model_cache_dir.exists():
        return model_name
    
    # Find the latest snapshot
    snapshots_dir = model_cache_dir / "snapshots"
    if not snapshots_dir.exists():
        return model_name
    
    # Get the latest snapshot directory
    snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
    if not snapshot_dirs:
        return model_name
    
    # Sort by modification time, take the latest
    latest_snapshot = max(snapshot_dirs, key=lambda x: x.stat().st_mtime)
    
    # Check if snapshot directory contains necessary files
    required_files = ["config.json", "model.safetensors", "tokenizer.json"]
    if all((latest_snapshot / f).exists() for f in required_files):
        logger.info(f"Found local cache: {latest_snapshot}")
        return str(latest_snapshot)
    
    return model_name


def load_model_with_cache(model_name: str, model_class=None, **kwargs):
    """
    Unified model loading function, prioritize local cache, allow network download if cache is incomplete
    
    Args:
        model_name: Model name or path
        model_class: Model class (e.g., AutoModel, AutoModelForCausalLM, etc.)
        **kwargs: Other parameters passed to from_pretrained
    
    Returns:
        Loaded model object
    """
    if model_class is None:
        from transformers import AutoModel
        model_class = AutoModel
    # Always set trust_remote_code=True unless explicitly set
    if "trust_remote_code" not in kwargs:
        kwargs["trust_remote_code"] = True
    # Try to get local cache path
    local_path = get_local_cache_path(model_name)
    try:
        # Prioritize using local cache
        logger.info(f"Attempting to load model from local cache: {local_path}")
        return model_class.from_pretrained(local_path, local_files_only=True, **kwargs)
    except Exception as e:
        logger.warning(f"Local cache incomplete, attempting to load from network: {model_name}")
        return model_class.from_pretrained(model_name, **kwargs)


def load_tokenizer_with_cache(model_name: str, **kwargs):
    """
    Unified tokenizer loading function, prioritize local cache, allow network download if cache is incomplete
    
    Args:
        model_name: Model name or path
        **kwargs: Other parameters passed to from_pretrained
    
    Returns:
        Loaded tokenizer object
    """
    from transformers import AutoTokenizer
    # Always set trust_remote_code=True unless explicitly set
    if "trust_remote_code" not in kwargs:
        kwargs["trust_remote_code"] = True
    # Try to get local cache path
    local_path = get_local_cache_path(model_name)
    try:
        # Prioritize using local cache
        logger.info(f"Attempting to load tokenizer from local cache: {local_path}")
        return AutoTokenizer.from_pretrained(local_path, local_files_only=True, **kwargs)
    except Exception as e:
        logger.warning(f"Local cache incomplete, attempting to load tokenizer from network: {model_name}")
        return AutoTokenizer.from_pretrained(model_name, **kwargs)


def get_disk_usage(path: str = "/") -> Tuple[int, int, int]:
    """
    Get disk usage information for a path.
    
    Args:
        path: Path to check disk usage for
        
    Returns:
        Tuple of (total_bytes, used_bytes, free_bytes)
    """
    try:
        total, used, free = shutil.disk_usage(path)
        return total, used, free
    except Exception as e:
        logger.warning(f"Could not get disk usage for {path}: {e}")
        return 0, 0, 0


def format_bytes(bytes_value: int) -> str:
    """Format bytes in human readable format with GiB/MiB units"""
    if bytes_value == 0:
        return "0B"
    
    # Use binary prefixes (GiB, MiB) for consistency with the requirement
    size_names = ["B", "KiB", "MiB", "GiB", "TiB"]
    i = 0
    while bytes_value >= 1024 and i < len(size_names) - 1:
        bytes_value = bytes_value / 1024.0
        i += 1
    
    return f"{bytes_value:.1f}{size_names[i]}"


def check_disk_space_safety(
    required_bytes: int, 
    safety_margin_gib: float = 5.0,
    path: str = "/"
) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if there's enough disk space with safety margin.
    
    Args:
        required_bytes: Required space in bytes
        safety_margin_gib: Safety margin in GiB (default: 5.0)
        path: Path to check disk usage for
        
    Returns:
        Tuple of (has_enough_space, info_dict)
    """
    total, used, free = get_disk_usage(path)
    safety_margin_bytes = int(safety_margin_gib * 1024**3)
    
    # Check if we have enough space for the operation
    has_enough_for_operation = free >= required_bytes
    
    # Check if we'll have enough safety margin after the operation
    remaining_after_operation = free - required_bytes
    has_safety_margin = remaining_after_operation >= safety_margin_bytes
    
    info = {
        "total_bytes": total,
        "used_bytes": used,
        "free_bytes": free,
        "required_bytes": required_bytes,
        "safety_margin_bytes": safety_margin_bytes,
        "remaining_after_operation": remaining_after_operation,
        "has_enough_for_operation": has_enough_for_operation,
        "has_safety_margin": has_safety_margin,
        "formatted": {
            "total": format_bytes(total),
            "used": format_bytes(used),
            "free": format_bytes(free),
            "required": format_bytes(required_bytes),
            "safety_margin": format_bytes(safety_margin_bytes),
            "remaining_after": format_bytes(remaining_after_operation)
        }
    }
    
    return has_enough_for_operation and has_safety_margin, info


def estimate_model_size(
    model_path: str, 
    output_format: str, 
    quantization: Optional[str] = None
) -> int:
    """
    Estimate the size of the output model in bytes.
    
    Args:
        model_path: Input model path or name
        output_format: Target output format
        quantization: Quantization type if applicable
        
    Returns:
        Estimated size in bytes
    """
    try:
        # Check if it's a local file
        if os.path.exists(model_path):
            if os.path.isfile(model_path):
                base_size = os.path.getsize(model_path)
            else:
                # Directory - sum all files
                base_size = get_directory_size(model_path)
        else:
            # Estimate based on model name
            lower_name = model_path.lower()
            if "bert-base" in lower_name:
                base_size = 420 * 1024**2  # 420MB
            elif "llama-2-7b" in lower_name or "llama2-7b" in lower_name:
                base_size = 13 * 1024**3  # 13GB
            elif "llama-2-13b" in lower_name or "llama2-13b" in lower_name:
                base_size = 24 * 1024**3  # 24GB
            elif "llama-2-70b" in lower_name or "llama2-70b" in lower_name:
                base_size = 130 * 1024**3  # 130GB
            elif "gpt2" in lower_name:
                base_size = 500 * 1024**2  # 500MB
            elif "opt-125m" in lower_name:
                base_size = 250 * 1024**2  # 250MB
            elif "opt-350m" in lower_name:
                base_size = 700 * 1024**2  # 700MB
            elif "opt-1.3b" in lower_name:
                base_size = 2.5 * 1024**3  # 2.5GB
            elif "opt-6.7b" in lower_name:
                base_size = 13 * 1024**3  # 13GB
            else:
                # Default estimate based on common model sizes
                base_size = 1 * 1024**3  # 1GB
        
        # Apply format-specific multipliers
        format_multipliers = {
            "onnx": 1.2,  # ONNX can be slightly larger
            "gguf": 0.8,  # GGUF is typically smaller
            "torchscript": 1.0,  # Similar to original
            "fp16": 0.5,  # Half precision
            "safetensors": 1.0,  # Similar to original
            "mlx": 0.8,  # MLX optimized
        }
        
        # Apply quantization multipliers
        quant_multipliers = {
            "4bit": 0.25,
            "8bit": 0.5,
            "q4_k_m": 0.25,
            "q8_0": 0.5,
            "q5_k_m": 0.31,
            "q4_0": 0.25,
            "q4_1": 0.25,
        }
        
        multiplier = format_multipliers.get(output_format, 1.0)
        if quantization and quantization in quant_multipliers:
            multiplier *= quant_multipliers[quantization]
        
        estimated_size = int(base_size * multiplier)
        
        # Add some buffer for temporary files during conversion
        conversion_buffer = int(estimated_size * 0.5)  # 50% buffer for temp files
        
        return estimated_size + conversion_buffer
        
    except Exception as e:
        logger.warning(f"Could not estimate model size: {e}")
        # Return a conservative estimate
        return 2 * 1024**3  # 2GB default


def prompt_user_confirmation_low_space(space_info: Dict[str, Any]) -> bool:
    """
    Prompt user for confirmation when disk space is low.
    
    Args:
        space_info: Disk space information from check_disk_space_safety
        
    Returns:
        True if user confirms, False otherwise
    """
    formatted = space_info["formatted"]
    
    warning_text = Text()
    warning_text.append("⚠️  ", style="bold yellow")
    warning_text.append("LOW DISK SPACE WARNING\n\n", style="bold red")
    
    warning_text.append("Current disk space:\n", style="bold")
    warning_text.append(f"  Free: {formatted['free']}\n")
    warning_text.append(f"  Required: {formatted['required']}\n")
    warning_text.append(f"  Safety margin: {formatted['safety_margin']}\n")
    warning_text.append(f"  Remaining after operation: {formatted['remaining_after']}\n\n")
    
    warning_text.append("After this operation, you will have less than the recommended ", style="yellow")
    warning_text.append("5 GiB", style="bold yellow")
    warning_text.append(" safety margin.\n\n", style="yellow")
    
    warning_text.append("This could cause system instability or prevent other applications from working properly.\n\n")
    
    warning_text.append("Do you want to continue anyway? Type 'yes' to confirm: ", style="bold")
    
    panel = Panel(
        warning_text,
        title="[bold red]Disk Space Warning[/bold red]",
        border_style="red"
    )
    
    console.print(panel)
    
    try:
        user_input = input().strip().lower()
        return user_input == "yes"
    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]Operation cancelled by user.[/yellow]")
        return False


def check_and_handle_disk_space(
    model_path: str,
    output_format: str,
    quantization: Optional[str] = None,
    safety_margin_gib: float = 5.0,
    path: str = "/"
) -> bool:
    """
    Check disk space and handle low space scenarios.
    
    Args:
        model_path: Input model path or name
        output_format: Target output format
        quantization: Quantization type if applicable
        safety_margin_gib: Safety margin in GiB
        path: Path to check disk usage for
        
    Returns:
        True if operation should proceed, False if aborted
    """
    # Estimate required space
    required_bytes = estimate_model_size(model_path, output_format, quantization)
    
    # Check disk space
    has_enough_space, space_info = check_disk_space_safety(
        required_bytes, safety_margin_gib, path
    )
    
    if has_enough_space:
        # Log disk space info for transparency
        formatted = space_info["formatted"]
        console.print(f"[green]✓[/green] Sufficient disk space available:")
        console.print(f"  Free: {formatted['free']} | Required: {formatted['required']} | Safety margin: {formatted['safety_margin']}")
        return True
    
    # Check if we have enough space for the operation but not enough safety margin
    if space_info["has_enough_for_operation"] and not space_info["has_safety_margin"]:
        return prompt_user_confirmation_low_space(space_info)
    
    # Not enough space even for the operation
    formatted = space_info["formatted"]
    error_text = Text()
    error_text.append("❌ ", style="bold red")
    error_text.append("INSUFFICIENT DISK SPACE\n\n", style="bold red")
    error_text.append("Current disk space:\n", style="bold")
    error_text.append(f"  Free: {formatted['free']}\n")
    error_text.append(f"  Required: {formatted['required']}\n")
    error_text.append(f"  Shortage: {format_bytes(required_bytes - space_info['free_bytes'])}\n\n")
    error_text.append("Please free up disk space and try again.", style="yellow")
    
    panel = Panel(
        error_text,
        title="[bold red]Disk Space Error[/bold red]",
        border_style="red"
    )
    
    console.print(panel)
    return False
