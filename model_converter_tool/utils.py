"""
Utility functions for model conversion
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

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
        size_bytes /= 1024.0
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
    temp_dir = Path("temp") / f"conv_{os.getpid()}_{int(os.time.time())}"
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


def create_dummy_model(
    output_dir: str,
    hidden_size: int = 4096,
    num_hidden_layers: int = 32,
    num_attention_heads: int = 32,
    vocab_size: int = 32000,
    max_position_embeddings: int = 2048,
    model_type: str = "llama",
    intermediate_size: int = None,
    **kwargs,
):
    """
    生成 HuggingFace 兼容的 dummy model（以 Llama 架构为例）。
    """
    os.makedirs(output_dir, exist_ok=True)
    if intermediate_size is None:
        intermediate_size = hidden_size * 4

    # 1. 生成 config.json
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": model_type,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_attention_heads": num_attention_heads,
        "num_hidden_layers": num_hidden_layers,
        "vocab_size": vocab_size,
        "max_position_embeddings": max_position_embeddings,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "torch_dtype": "float16",
        **kwargs,
    }
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # 2. 生成随机权重的 state_dict - Create a more realistic Llama-like model
    try:
        # Try to use actual LlamaForCausalLM for compatibility
        from transformers import LlamaConfig, LlamaForCausalLM

        llama_config = LlamaConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            intermediate_size=config["intermediate_size"],
            num_hidden_layers=config["num_hidden_layers"],
            num_attention_heads=config["num_attention_heads"],
            max_position_embeddings=config["max_position_embeddings"],
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            bos_token_id=config["bos_token_id"],
            eos_token_id=config["eos_token_id"],
            pad_token_id=config["pad_token_id"],
            torch_dtype="float16",
        )

        # Create the model with small random weights
        model = LlamaForCausalLM(llama_config)

        # Initialize with smaller weights for faster testing
        for name, param in model.named_parameters():
            if "weight" in name:
                param.data.normal_(mean=0.0, std=0.02)
            elif "bias" in name:
                param.data.zero_()

        # Save the model
        model.save_pretrained(output_dir, safe_serialization=False)

    except Exception as e:
        logger.warning(f"Failed to create LlamaForCausalLM: {e}, falling back to simple model")

        # Fallback: create minimal state dict manually
        state_dict = {
            "model.embed_tokens.weight": torch.randn(config["vocab_size"], config["hidden_size"]) * 0.02,
            "lm_head.weight": torch.randn(config["vocab_size"], config["hidden_size"]) * 0.02,
        }

        # Add transformer layers
        for layer_idx in range(config["num_hidden_layers"]):
            prefix = f"model.layers.{layer_idx}"
            state_dict.update(
                {
                    f"{prefix}.self_attn.q_proj.weight": torch.randn(config["hidden_size"], config["hidden_size"])
                    * 0.02,
                    f"{prefix}.self_attn.k_proj.weight": torch.randn(config["hidden_size"], config["hidden_size"])
                    * 0.02,
                    f"{prefix}.self_attn.v_proj.weight": torch.randn(config["hidden_size"], config["hidden_size"])
                    * 0.02,
                    f"{prefix}.self_attn.o_proj.weight": torch.randn(config["hidden_size"], config["hidden_size"])
                    * 0.02,
                    f"{prefix}.mlp.gate_proj.weight": torch.randn(config["intermediate_size"], config["hidden_size"])
                    * 0.02,
                    f"{prefix}.mlp.up_proj.weight": torch.randn(config["intermediate_size"], config["hidden_size"])
                    * 0.02,
                    f"{prefix}.mlp.down_proj.weight": torch.randn(config["hidden_size"], config["intermediate_size"])
                    * 0.02,
                    f"{prefix}.input_layernorm.weight": torch.ones(config["hidden_size"]),
                    f"{prefix}.post_attention_layernorm.weight": torch.ones(config["hidden_size"]),
                }
            )

        # Add final norm
        state_dict["model.norm.weight"] = torch.ones(config["hidden_size"])

        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))

    # 3. Create simple tokenizer files manually (avoid network dependency)
    # Create basic tokenizer config
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "model_max_length": max_position_embeddings,
    }
    with open(os.path.join(output_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, indent=2)

    # Create simple vocabulary
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
    for i in range(4, min(vocab_size, 1000)):  # Limit vocab size for CI
        vocab[f"token{i}"] = i

    with open(os.path.join(output_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)

    # Create simplified tokenizer.json
    tokenizer_data = {
        "version": "1.0",
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"},
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "added_tokens": [
            {
                "id": 0,
                "content": "<pad>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
            {
                "id": 1,
                "content": "<s>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
            {
                "id": 2,
                "content": "</s>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
            {
                "id": 3,
                "content": "<unk>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
        ],
    }
    with open(os.path.join(output_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(tokenizer_data, f, indent=2)

    print(f"✅ Dummy model generated at {output_dir}")
