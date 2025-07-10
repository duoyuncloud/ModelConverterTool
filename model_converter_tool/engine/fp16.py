import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

def convert_to_fp16(
    model: Any,
    tokenizer: Any,
    model_name: str,
    output_path: str,
    model_type: str,
    device: str
) -> tuple:
    """
    Export model to FP16 safetensors format.
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
        import torch
        from safetensors.torch import save_file
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            model = model.half()
            has_shared = False
            if hasattr(model, "lm_head") and hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
                try:
                    if model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr():
                        has_shared = True
                except Exception:
                    pass
            if has_shared:
                logger.info("Detected shared weights, using save_pretrained for safe serialization.")
                model.save_pretrained(str(output_dir), safe_serialization=True)
            else:
                state_dict = model.state_dict()
                fp16_state_dict = {}
                for key, value in state_dict.items():
                    if value.dtype == torch.float32:
                        fp16_state_dict[key] = value.half()
                    else:
                        fp16_state_dict[key] = value
                save_file(fp16_state_dict, output_dir / "model.safetensors")
            logger.info(f"FP16 conversion completed: {output_dir}")
            return True, None
        except Exception as e:
            logger.error(f"FP16 conversion failed: {e}")
            return False, None
    except Exception as e:
        logger.error(f"FP16 conversion error: {e}")
        return False, None

def validate_fp16_file(fp16_dir: Path, _: Any) -> bool:
    """
    Validate FP16 safetensors file validity.
    Args:
        fp16_dir: Output directory
    Returns:
        bool: Whether valid
    """
    try:
        # Consider valid as long as directory exists and contains model.safetensors or config.json
        if not fp16_dir.exists():
            logger.warning(f"FP16 output dir does not exist: {fp16_dir}")
            return False
        if (fp16_dir / "model.safetensors").exists() or (fp16_dir / "config.json").exists():
            return True
        logger.warning(f"FP16 output missing model.safetensors and config.json: {fp16_dir}")
        return False
    except Exception as e:
        logger.warning(f"FP16 validation error: {e}")
        return False 