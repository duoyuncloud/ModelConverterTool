import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

def convert_to_torchscript(
    model: Any,
    tokenizer: Any,
    model_name: str,
    output_path: str,
    model_type: str,
    device: str
) -> tuple:
    """
    将模型导出为 TorchScript 格式。
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
        torchscript_file = Path(output_path)
        torchscript_file.parent.mkdir(parents=True, exist_ok=True)
        export_success = False
        # Step 1: Try torch.jit.script
        try:
            logger.info("Attempting torch.jit.script...")
            model.eval()
            if model_type == "text-generation":
                class TextGenWrapper(torch.nn.Module):
                    def __init__(self, model):
                        super().__init__()
                        self.model = model
                    def forward(self, input_ids, attention_mask=None):
                        if attention_mask is None:
                            attention_mask = torch.ones_like(input_ids)
                        return self.model(input_ids, attention_mask=attention_mask)
                wrapped_model = TextGenWrapper(model)
                scripted_model = torch.jit.script(wrapped_model)
            else:
                scripted_model = torch.jit.script(model)
            scripted_model.save(str(torchscript_file))
            export_success = True
            logger.info("TorchScript script export successful")
            return True, None
        except Exception as e:
            logger.warning(f"TorchScript script export failed: {e}")
        # Step 2: Try torch.jit.trace
        try:
            logger.info("Attempting torch.jit.trace...")
            model.eval()
            if model_type == "image-classification":
                dummy_input = torch.randn(1, 3, 224, 224)
            elif model_type == "text-generation":
                dummy_input = torch.randint(0, min(tokenizer.vocab_size, 1000), (1, 10))
            else:
                dummy_input = torch.randint(0, min(tokenizer.vocab_size, 1000), (1, 32))
                if hasattr(model, "forward") and "attention_mask" in str(model.forward.__code__.co_varnames):
                    dummy_mask = torch.ones_like(dummy_input)
                    dummy_input = (dummy_input, dummy_mask)
            traced_model = torch.jit.trace(model, dummy_input, strict=False)
            traced_model.save(str(torchscript_file))
            export_success = True
            logger.info("TorchScript trace export successful")
            return True, None
        except Exception as e:
            logger.warning(f"TorchScript trace export failed: {e}")
        logger.error("All TorchScript export methods failed")
        return False, None
    except Exception as e:
        logger.error(f"TorchScript conversion error: {e}")
        return False, None

def validate_torchscript_file(torchscript_file: Path, _: any) -> bool:
    try:
        import torch
        if not torchscript_file.exists() or torchscript_file.stat().st_size < 100:
            return False
        model = torch.jit.load(str(torchscript_file))
        model.eval()
        dummy_input = torch.zeros((1, 8), dtype=torch.long)
        with torch.no_grad():
            try:
                _ = model(dummy_input, torch.ones((1, 8), dtype=torch.long))
            except Exception:
                _ = model(dummy_input)
        return True
    except Exception:
        return False 