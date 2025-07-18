import logging
from pathlib import Path
from typing import Any, Optional
from model_converter_tool.utils import auto_load_model_and_tokenizer
import os

logger = logging.getLogger(__name__)

def convert_to_safetensors(
    model: Any,
    tokenizer: Any,
    model_name: str,
    output_path: str,
    model_type: str,
    device: str,
    dtype: str = None
) -> tuple:
    """
    Save model in safetensors format, with optional precision control.
    Args:
        model: Loaded model object (optional)
        tokenizer: Loaded tokenizer object (optional)
        model_name: Source model name or path
        output_path: Output file path
        model_type: Model type
        device: Device
        dtype: Precision for weights (e.g., 'fp16', 'fp32')
    Returns:
        (success: bool, extra_info: dict or error string)
    """
    try:
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Robust model auto-loading
        model, tokenizer = auto_load_model_and_tokenizer(model, tokenizer, model_name, model_type)
        try:
            if dtype == "fp16":
                model = model.half()
            elif dtype == "fp32":
                model = model.float()
            # Assert model and output_dir are valid before saving
            assert model is not None, "Model is None before saving!"
            assert output_dir is not None, "Output directory is None!"
            model.save_pretrained(str(output_dir), safe_serialization=True)
            # Only save tokenizer if it exists
            if tokenizer is not None:
                tokenizer.save_pretrained(str(output_dir))
            logger.info(f"Safetensors conversion completed: {output_dir} (dtype={dtype or 'default'})")
            return True, None
        except Exception as e:
            logger.error(f"Safetensors conversion failed: {e}")
            return False, str(e)
    except Exception as e:
        logger.error(f"Safetensors conversion error: {e}")
        return False, str(e)

def validate_safetensors_file(path: str, *args, **kwargs) -> bool:
    """
    Static validation for SafeTensors files. Accepts either a file or directory path.
    If a directory is given, looks for 'model.safetensors' inside it.
    Returns True if the file passes static validation, False otherwise.
    """
    if not os.path.exists(path):
        return False
    # If path is a directory, look for 'model.safetensors' inside
    if os.path.isdir(path):
        safetensors_file = os.path.join(path, "model.safetensors")
        if not os.path.exists(safetensors_file):
            return False
        path = safetensors_file
    try:
        from safetensors.torch import load_file
        tensors = load_file(path)
        return bool(tensors)
    except ImportError:
        # safetensors not installed
        return False
    except Exception:
        return False

def can_infer_safetensors_file(path: str, *args, **kwargs) -> bool:
    """
    Dynamic check for SafeTensors files. Loads the model and tokenizer and runs a real dummy inference.
    Returns True if inference is possible, False otherwise.
    """
    try:
        import os
        from transformers import AutoModel, AutoTokenizer
        # If path is a directory, use it directly; if file, use its parent
        if os.path.isdir(path):
            model_dir = path
        else:
            model_dir = os.path.dirname(path)
        model = AutoModel.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        dummy = tokenizer("Hello world!", return_tensors="pt")
        _ = model(**dummy)
        return True
    except Exception:
        return False 