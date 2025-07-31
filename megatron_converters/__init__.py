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

# from .loader_llama2_hf import load_checkpoint as llama2_hf_load_checkpoint
# from .loader_llama_mistral import load_checkpoint as llama_mistral_load_checkpoint
from .loader_minicpm_hf import load_checkpoint as minicpm_load_checkpoint
import queue as py_queue


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
