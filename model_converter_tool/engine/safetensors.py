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
    device: str
) -> tuple:
    """
    将模型保存为 safetensors 格式。
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
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            model.save_pretrained(str(output_dir), safe_serialization=True)
            if tokenizer:
                tokenizer.save_pretrained(str(output_dir))
            logger.info(f"Safetensors conversion completed: {output_dir}")
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
        model = AutoModel.from_pretrained(str(st_dir))
        tokenizer = AutoTokenizer.from_pretrained(str(st_dir))
        inputs = tokenizer("hello world", return_tensors="pt")
        with torch.no_grad():
            _ = model(**inputs)
        return True
    except Exception:
        return False 