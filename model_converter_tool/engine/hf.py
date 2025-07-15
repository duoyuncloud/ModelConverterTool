import logging
from pathlib import Path
from typing import Any, Optional
from model_converter_tool.utils import auto_load_model_and_tokenizer

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

def validate_hf_file(hf_dir, _=None):
    try:
        from transformers import AutoModel, AutoTokenizer
        model = AutoModel.from_pretrained(hf_dir)
        tokenizer = AutoTokenizer.from_pretrained(hf_dir)
        inputs = tokenizer("Test prompt", return_tensors="pt")
        _ = model(**inputs)
        return True
    except Exception as e:
        import logging, traceback
        logging.getLogger(__name__).error(f"HuggingFace validation failed: {e}")
        traceback.print_exc()
        return False 