"""
Unified interface for Llama and MiniCPM Megatron<->HF conversion in ModelConverterTool.

Usage:
    from megatron_converters import convert_llama, convert_minicpm
    args = ...  # your argument object
    convert_llama(args, direction='megatron2hf')
    convert_minicpm(args, direction='hf2megatron')

Arguments:
    args: An object with necessary attributes for the loader (e.g., load_dir, save_dir, etc.)
    direction: 'megatron2hf' or 'hf2megatron' (default: 'megatron2hf')

You can extend this interface to support more loaders or directions as needed.
"""

from .loader_llama2 import load_checkpoint as llama2_load_checkpoint
from .loader_minicpm_hf import load_checkpoint as minicpm_load_checkpoint
import queue as py_queue

# Tensor Parallel and Pipeline Parallel converters
from .tp_pp_converter import (
    TensorParallelConverter,
    convert_minicpm_megatron_to_hf_tp_pp,
    convert_llama_megatron_to_hf_tp_pp,
    convert_8b_minicpm_megatron_to_hf,
    convert_3b_minicpm_megatron_to_hf,
    convert_7b_llama_megatron_to_hf,
)

# Smart converter
from .smart_converter import (
    SmartConverter,
    smart_convert_megatron_to_hf,
    smart_convert_hf_to_megatron,
    convert_minicpm_8b,
    convert_minicpm_3b,
    convert_llama_7b,
    convert_minicpm4_8b,
)

# Direct checkpoint converters
from .ckpt_to_hf_minicpm_with_tp_pp import convert_minicpm_megatron_to_hf_tp_pp as ckpt_convert_minicpm_tp_pp

# Distributed checkpoint converters
try:
    from .dist_ckpt_to_hf_minicpm import convert_minicpm_megatron_to_hf as dist_ckpt_to_hf_minicpm
    from .dist_ckpt_to_hf_minicpm4 import convert_minicpm4_megatron_to_hf as dist_ckpt_to_hf_minicpm4
except ImportError:
    # These may not be available if Megatron is not installed
    dist_ckpt_to_hf_minicpm = None
    dist_ckpt_to_hf_minicpm4 = None

# HF to Megatron converters
from .hf_to_megatron_minicpm import convert_hf_to_megatron_minicpm
from .hf_to_megatron_minicpm4 import convert_hf_to_megatron_minicpm4

__all__ = [
    # Core converters
    "TensorParallelConverter",
    "SmartConverter",
    # TP/PP converters
    "convert_minicpm_megatron_to_hf_tp_pp",
    "convert_llama_megatron_to_hf_tp_pp", 
    "convert_8b_minicpm_megatron_to_hf",
    "convert_3b_minicpm_megatron_to_hf",
    "convert_7b_llama_megatron_to_hf",
    # Smart converters
    "smart_convert_megatron_to_hf",
    "smart_convert_hf_to_megatron",
    "convert_minicpm_8b",
    "convert_minicpm_3b",
    "convert_llama_7b",
    "convert_minicpm4_8b",
    # Direct checkpoint converters
    "ckpt_convert_minicpm_tp_pp",
    # Distributed checkpoint converters
    "dist_ckpt_to_hf_minicpm",
    "dist_ckpt_to_hf_minicpm4",
    # HF to Megatron converters
    "convert_hf_to_megatron_minicpm",
    "convert_hf_to_megatron_minicpm4",
    # Legacy loaders (for backward compatibility)
    "llama2_load_checkpoint",
    "minicpm_load_checkpoint",
]


def convert_llama(args, direction="megatron2hf"):
    """
    Convert Llama/Llama2/Mistral model weights between Megatron and HuggingFace formats.
    Args:
        args: Argument object for the loader.
        direction: 'megatron2hf' or 'hf2megatron'.
    """
    # You can add logic here to select different loaders based on args or direction
    # For now, use llama2 loader as default
    # The loader expects (queue, args), so we provide a dummy queue
    dummy_queue = py_queue.Queue()
    return llama2_load_checkpoint(dummy_queue, args)


def convert_minicpm(args, direction="megatron2hf"):
    """
    Convert MiniCPM model weights between Megatron and HuggingFace formats.
    Args:
        args: Argument object for the loader.
        direction: 'megatron2hf' or 'hf2megatron'.
    """
    # The loader expects (queue, args, direction), so we provide a dummy queue
    dummy_queue = py_queue.Queue()
    return minicpm_load_checkpoint(dummy_queue, args, direction)
