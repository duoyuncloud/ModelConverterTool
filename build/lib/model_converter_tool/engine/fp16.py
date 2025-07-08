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
    将模型导出为 FP16 safetensors 格式。
    Args:
        model: 已加载的模型对象
        tokenizer: 已加载的分词器对象
        model_name: 源模型名称或路径
        output_path: 输出文件路径
        model_type: 模型类型
        device: 设备
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
    验证 FP16 safetensors 文件有效性。
    Args:
        fp16_dir: 输出目录
    Returns:
        bool: 是否有效
    """
    try:
        # 只要目录存在且包含 model.safetensors 或 config.json 即认为有效
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