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
import re
import textwrap

logger = logging.getLogger(__name__)
console = Console()


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


def ensure_output_directory(output_path: str) -> str:
    """Ensure output directory exists and return the path"""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_path)


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


def load_model_with_cache(model_name: str, model_class=None, fake_weight: bool = False, fake_weight_shape_dict: dict = None, **kwargs):
    """
    Load a model from cache or disk. If fake_weight is True, generate a model with the correct architecture and all weights set to zero or custom shapes.
    fake_weight_shape_dict: Optional dict specifying custom shapes for each weight tensor.
    """
    if fake_weight:
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
        import torch
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        # Remove quantization_config if present to avoid loading errors with fake weights
        if hasattr(config, 'quantization_config'):
            delattr(config, 'quantization_config')
        if hasattr(config, 'to_dict') and 'quantization_config' in config.to_dict():
            # Some configs may keep it in the dict representation
            config_dict = config.to_dict()
            config_dict.pop('quantization_config', None)
            # Rebuild config from dict to ensure it's clean
            config = type(config).from_dict(config_dict)
        if model_class is None:
            model_type = getattr(config, 'model_type', None)
            if model_type is not None and ("qwen" in model_type):
                model_class = AutoModelForCausalLM
            else:
                model_class = AutoModel
        # Generate the model with all weights set to zero (default shapes)
        model = generate_fake_model(config, model_class, fake_weight_shape_dict)
        return model
    # Detect Qwen model type and use correct class
    if model_class is None:
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
        try:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            model_type = getattr(config, 'model_type', None)
            if model_type is not None and ("qwen" in model_type):
                model_class = AutoModelForCausalLM
            else:
                model_class = AutoModel
        except Exception:
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


def ansi_safe_wrap(text: str, width: int) -> str:
    """
    Wrap text to a given width, preserving ANSI color codes and word boundaries.
    Args:
        text: The input string (may contain ANSI codes)
        width: The max line width
    Returns:
        Wrapped string
    """
    # Regex to match ANSI escape sequences
    ansi_escape = re.compile(r"(\x1b\[[0-9;]*[a-zA-Z])")
    # Split text into ANSI and non-ANSI parts
    parts = ansi_escape.split(text)
    clean = ''
    ansi_stack = []
    lines = []
    for part in parts:
        if ansi_escape.match(part):
            ansi_stack.append(part)
            clean += part
        else:
            # Wrap the non-ANSI part
            wrapped = textwrap.wrap(part, width=width, replace_whitespace=False, drop_whitespace=False)
            for i, line in enumerate(wrapped):
                if i > 0:
                    # Reset ANSI codes at the start of each new line
                    lines.append(''.join(ansi_stack) + line)
                else:
                    clean += line
            if wrapped:
                clean = ''
    if clean:
        lines.append(clean)
    return '\n'.join(lines)


def auto_load_model_and_tokenizer(model, tokenizer, model_name, model_type):
    """
    Robustly load model and tokenizer if not provided. Returns (model, tokenizer).
    Uses AutoModelForCausalLM for text-generation/causal/lm/generation types, else AutoModel.
    If the specified model_type is not supported, fallback to 'auto'.
    """
    from transformers import AutoModel, AutoModelForCausalLM
    from model_converter_tool.utils import load_model_with_cache, load_tokenizer_with_cache
    def _load(model_type_to_use):
        if model is None:
            if model_type_to_use and any(x in model_type_to_use for x in ("causal", "lm", "generation", "text-generation")):
                loaded_model = load_model_with_cache(model_name, AutoModelForCausalLM)
            else:
                loaded_model = load_model_with_cache(model_name, AutoModel)
        else:
            loaded_model = model
        loaded_tokenizer = tokenizer or load_tokenizer_with_cache(model_name)
        return loaded_model, loaded_tokenizer
    try:
        # Try with user-specified model_type first
        return _load(model_type)
    except Exception as e:
        # If model_type is not 'auto', fallback to 'auto' on failure
        if model_type != "auto":
            try:
                return _load("auto")
            except Exception:
                # Raise the original error if fallback also fails
                raise e
        else:
            raise


def get_calibration_dataset(use_large_calibration, tag="AWQ"):
    """
    Returns a calibration dataset (list of strings) for quantization. If use_large_calibration is True, attempts to load openwebtext, else uses built-in samples.
    Tag is used for logging (e.g., 'AWQ', 'GPTQ').
    """
    import logging
    logger = logging.getLogger(__name__)
    if use_large_calibration:
        try:
            from datasets import load_dataset
            ds = load_dataset("openwebtext", split="train", trust_remote_code=True)
            calibration_dataset = [x["text"] for x in ds.select(range(1000)) if len(x["text"].split()) > 32]
            if len(calibration_dataset) < 1000:
                calibration_dataset += ["The quick brown fox jumps over the lazy dog."] * (1000 - len(calibration_dataset))
            logger.info(f"[{tag}] Using HuggingFace openwebtext sampling {len(calibration_dataset)} high-quality calibration texts")
            return calibration_dataset
        except Exception as e:
            logger.warning(f"[{tag}] Failed to load high-quality calibration set, falling back to built-in samples: {e}")
            return [
                "The quick brown fox jumps over the lazy dog. " * 20,
                f"{tag} high-precision calibration sentence. " * 20,
                "This is a long calibration text for high-precision quantization. " * 20,
            ]
    else:
        return [
            "This is a much longer calibration sentence that should have more than ten tokens for the quantization process.",
            "Another example of a calibration sentence that is sufficiently long to pass the minimum length requirement for quantization.",
            "Quantization calibration requires sentences that are not too short, so this one is also long enough to be valid for the test."
        ]


def patch_quantization_config(config_path, bits, group_size, sym, desc):
    """
    Patch or create a config.json file with quantization_config for test compatibility.
    """
    import json
    from pathlib import Path
    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = {}
    config["quantization_config"] = {
        "bits": bits,
        "group_size": group_size,
        "sym": sym,
        "desc": desc,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def generate_fake_model(config, model_class, fake_weight_shape_dict: dict = None):
    """
    Generate a model instance with the given config and fill all weights with zeros.
    If fake_weight_shape_dict is provided, use it to set custom shapes for weights.
    Args:
        config: Model config object
        model_class: Model class (e.g., AutoModel, AutoModelForCausalLM)
        fake_weight_shape_dict: Optional dict of {param_name: shape}
    Returns:
        Model instance with all weights set to zero (default or custom shapes)
    """
    import torch
    model = model_class.from_config(config)
    for name, param in model.named_parameters():
        if param.requires_grad:
            if fake_weight_shape_dict and name in fake_weight_shape_dict:
                try:
                    shape = tuple(fake_weight_shape_dict[name])
                    if shape != tuple(param.shape):
                        print(f"[WARNING] Shape mismatch for {name}: config {shape}, model default {tuple(param.shape)}. Using config shape.")
                    param.data = torch.zeros(shape, dtype=param.dtype, device=param.device)
                except Exception as e:
                    print(f"[WARNING] Could not set fake weight for {name} with shape {fake_weight_shape_dict[name]}: {e}")
            else:
                torch.nn.init.zeros_(param)
    return model


def create_dummy_model(output_dir: str, **kwargs):
    """
    Generate a simple test dummy model for testing purposes only.
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


def patch_config_remove_quantization_config(output_dir):
    """
    Remove quantization_config field from all .json files in the given directory.
    This is used to ensure fake-weight outputs are always loadable by transformers.
    """
    import json
    from pathlib import Path
    output_dir = Path(output_dir)
    for json_file in output_dir.glob('*.json'):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            if "quantization_config" in config_data:
                del config_data["quantization_config"]
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(config_data, f, indent=2)
        except Exception as e:
            print(f"[WARNING] Failed to patch {json_file}: {e}")


def auto_complete_output_path(input_path, output_path, to_format):
    """
    Generate a standardized output directory path for model conversion outputs.
    - Uses the full base name of the input path (including .5B or similar) and appends the format.
    - If output_path is provided and is a file, converts it to a directory with the format suffix.
    - If output_path is a directory, uses it directly.
    - If output_path is omitted, uses './outputs/{base}_{to_format}'.
    """
    import os
    from pathlib import Path
    output_aliases = {"hf": "huggingface"}
    to_format = output_aliases.get(to_format.lower(), to_format.lower())
    file_exts = {
        'onnx': '.onnx',
        'gguf': '.gguf',
        'pt': '.pt',
        'torchscript': '.pt',
        'safetensors': '.safetensors',
        'fp16': '.safetensors',
    }
    base = os.path.basename(input_path)  # Always preserve full name, including .5B, etc.
    def to_dir_name(path, ext=None):
        p = Path(path)
        if ext and p.name.endswith(ext):
            return str(p.with_suffix('')) + f'_{to_format}'
        if p.suffix:
            return str(p.with_suffix('')) + f'_{to_format}'
        return str(p)
    if not output_path:
        return f'./outputs/{base}_{to_format}'
    if os.path.isdir(output_path):
        return output_path
    for ext in file_exts.values():
        if output_path.endswith(ext):
            return to_dir_name(output_path, ext)
    if not os.path.exists(output_path) and not output_path.endswith('/'):
        return to_dir_name(output_path)
    return output_path