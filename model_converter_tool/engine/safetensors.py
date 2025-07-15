import logging
from pathlib import Path
from typing import Any, Optional
from model_converter_tool.utils import auto_load_model_and_tokenizer

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
            model.save_pretrained(str(output_dir), safe_serialization=True)
            if tokenizer:
                tokenizer.save_pretrained(str(output_dir))
            logger.info(f"Safetensors conversion completed: {output_dir} (dtype={dtype or 'default'})")
            return True, None
        except Exception as e:
            logger.error(f"Safetensors conversion failed: {e}")
            return False, str(e)
    except Exception as e:
        logger.error(f"Safetensors conversion error: {e}")
        return False, str(e)

def validate_safetensors_file(safetensors_path):
    try:
        from safetensors.torch import load_file
        tensors = load_file(safetensors_path)
        for k, v in tensors.items():
            _ = v.shape
            break
        return True
    except Exception as e:
        import logging, traceback
        logging.getLogger(__name__).error(f"SafeTensors validation failed: {e}")
        traceback.print_exc()
        return False 