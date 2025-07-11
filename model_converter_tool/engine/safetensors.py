import logging
from pathlib import Path
from typing import Any, Optional

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
        model: Loaded model object
        tokenizer: Loaded tokenizer object
        model_name: Source model name or path
        output_path: Output file path
        model_type: Model type
        device: Device
        dtype: Precision for weights (e.g., 'fp16', 'fp32')
    Returns:
        (success: bool, extra_info: dict or None)
    """
    try:
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            if dtype == "fp16":
                model = model.half()
            elif dtype == "fp32":
                model = model.float()
            model.save_pretrained(str(output_dir), safe_serialization=True)
            if tokenizer:
                tokenizer.save_pretrained(str(output_dir))
            logger.info(f"Safetensors conversion completed: {output_dir} (dtype={dtype or 'default'})")
            return True, None
        except Exception as e:
            logger.error(f"Safetensors conversion failed: {e}")
            return False, None
    except Exception as e:
        logger.error(f"Safetensors conversion error: {e}")
        return False, None

def validate_safetensors_file(st_dir: Path, _: any) -> bool:
    try:
        if not st_dir.exists():
            return False
        from transformers import AutoModel, AutoTokenizer
        import torch
        from model_converter_tool.utils import load_model_with_cache, load_tokenizer_with_cache
        model = load_model_with_cache(str(st_dir), AutoModel)
        tokenizer = load_tokenizer_with_cache(str(st_dir))
        inputs = tokenizer("hello world", return_tensors="pt")
        with torch.no_grad():
            _ = model(**inputs)
        return True
    except Exception:
        return False 