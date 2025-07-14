import logging
from pathlib import Path
from typing import Any, Optional
from model_converter_tool.utils import auto_load_model_and_tokenizer

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
    Export model to TorchScript format.
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
            dummy_input = None
            dummy_mask = None
            if tokenizer is None:
                logger.error("Tokenizer is None, cannot proceed with TorchScript export.")
                return False, {"error": "Tokenizer is None, cannot proceed with TorchScript export."}
            if model_type == "image-classification":
                dummy_input = torch.randn(1, 3, 224, 224)
            elif model_type == "text-generation":
                vocab_size = getattr(tokenizer, "vocab_size", 30522)
                if vocab_size is None:
                    logger.warning("Tokenizer has no vocab_size, using default vocab_size=30522")
                    vocab_size = 30522
                dummy_input = torch.randint(0, min(vocab_size, 1000), (1, 10))
            elif model_type in ("text-classification", "auto"):
                vocab_size = getattr(tokenizer, "vocab_size", 30522)
                if vocab_size is None:
                    logger.warning("Tokenizer has no vocab_size, using default vocab_size=30522")
                    vocab_size = 30522
                dummy_input = torch.randint(0, min(vocab_size, 1000), (1, 8))
                dummy_mask = torch.ones_like(dummy_input)
                if hasattr(model, "forward") and "attention_mask" in str(model.forward.__code__.co_varnames):
                    dummy_input = (dummy_input, dummy_mask)
            else:
                vocab_size = getattr(tokenizer, "vocab_size", 30522)
                if vocab_size is None:
                    logger.warning("Tokenizer has no vocab_size, using default vocab_size=30522")
                    vocab_size = 30522
                dummy_input = torch.randint(0, min(vocab_size, 1000), (1, 32))
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