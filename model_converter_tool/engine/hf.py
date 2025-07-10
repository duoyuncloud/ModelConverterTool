import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

def convert_to_hf(
    model: Any,
    tokenizer: Any,
    model_name: str,
    output_path: str,
    model_type: str,
    device: str
) -> tuple:
    """
    Save model in HuggingFace native format.
    Args:
        model: Loaded model object
        tokenizer: Loaded tokenizer object
        model_name: Source model name or path
        output_path: Output file path
        model_type: Model type
        device: Device
    Returns:
        (success: bool, extra_info: dict or None)
    """
    try:
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

def validate_hf_file(hf_dir: Path, _: any) -> bool:
    try:
        if not hf_dir.exists():
            return False
        from transformers import AutoModel, AutoTokenizer
        import torch
        from model_converter_tool.utils import load_model_with_cache, load_tokenizer_with_cache
        model = load_model_with_cache(str(hf_dir), AutoModel)
        tokenizer = load_tokenizer_with_cache(str(hf_dir))
        inputs = tokenizer("hello world", return_tensors="pt")
        with torch.no_grad():
            _ = model(**inputs)
        return True
    except Exception:
        return False 