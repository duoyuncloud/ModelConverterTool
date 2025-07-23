import logging
import os
from pathlib import Path
from typing import Any
from model_converter_tool.utils import auto_load_model_and_tokenizer

logger = logging.getLogger(__name__)


def convert_to_hf(model: Any, tokenizer: Any, model_name: str, output_path: str, model_type: str, device: str) -> tuple:
    """
    Save model in HuggingFace native format.
    Args:
        model: Loaded model object (optional)
        tokenizer: Loaded tokenizer object (optional)
        model_name: Source model name or path
        output_path: Output file path
        model_type: Model type
        device: Device
    Returns:
        (success: bool, extra_info: dict or None)
    """
    try:
        # Robust model/tokenizer auto-loading
        model, tokenizer = auto_load_model_and_tokenizer(model, tokenizer, model_name, model_type)
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            model.save_pretrained(str(output_dir), safe_serialization=True)
            if tokenizer:
                tokenizer.save_pretrained(str(output_dir))
            logger.info(f"HF conversion completed: {output_dir}")
            return True, None
        except Exception as e:
            logger.error(f"HF conversion failed: {e}")
            return False, None
    except Exception as e:
        logger.error(f"HF conversion error: {e}")
        return False, None


def validate_hf_file(path: str, *args, **kwargs) -> bool:
    """
    Static validation for HuggingFace model directories. Checks if config.json exists and the model can be loaded by transformers.AutoModel.
    Returns True if the directory passes static validation, False otherwise.
    """
    if not os.path.exists(path) or not os.path.isdir(path):
        return False
    config_path = os.path.join(path, "config.json")
    if not os.path.exists(config_path):
        return False
    try:
        from transformers import AutoModel

        _ = AutoModel.from_pretrained(path)
        return True
    except ImportError:
        # transformers not installed
        return False
    except Exception:
        return False


def can_infer_hf_file(path: str, *args, **kwargs):
    """
    Dynamic check for HuggingFace model directories. Loads the model and tokenizer and runs a real dummy inference.
    Returns (True, None) if inference is possible, (False, error_message) otherwise.
    """
    try:
        from transformers import AutoModel, AutoTokenizer

        model = AutoModel.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        dummy = tokenizer("Hello world!", return_tensors="pt")
        _ = model(**dummy)
        return True, None
    except ImportError:
        return False, "transformers not installed"
    except Exception as e:
        return False, str(e)
