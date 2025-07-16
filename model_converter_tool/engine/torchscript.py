import logging
import os
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
        # User-friendly error for unsupported models
        unsupported = ["gpt2", "llama", "mistral", "qwen", "mixtral", "vit", "t5", "bart", "clip"]
        lower_name = model_name.lower()
        if any(u in lower_name for u in unsupported):
            logger.error(f"TorchScript export is not fully supported for model '{model_name}'.\n"
                         f"This model type is known to have limited or unstable TorchScript support due to dynamic control flow, *args/**kwargs, or tied weights.\n"
                         f"For best results, use BERT, DistilBERT, or ResNet models.\n"
                         f"Original error: {e}")
            return False, {"error": f"TorchScript export is not fully supported for model '{model_name}'. For best results, use BERT, DistilBERT, or ResNet. See logs for details."}
        logger.error("All TorchScript export methods failed")
        return False, None
    except Exception as e:
        logger.error(f"TorchScript conversion error: {e}")
        return False, None

def validate_torchscript_file(path: str, *args, **kwargs) -> bool:
    """
    Static validation for TorchScript files. Checks if the file exists and can be loaded by torch.jit.load.
    Returns True if the file passes static validation, False otherwise.
    """
    if not os.path.exists(path):
        return False
    try:
        import torch
        _ = torch.jit.load(path, map_location='cpu')
        return True
    except ImportError:
        # torch not installed
        return False
    except Exception:
        return False

def can_infer_torchscript_file(path: str, *args, **kwargs) -> bool:
    """
    Dynamic check for TorchScript files. Loads the model and runs a real dummy inference using torch.jit.
    Tries to infer the model type and construct realistic dummy inputs for BERT, GPT-2, vision, seq2seq, CLIP, and other HuggingFace models.
    Returns True if inference is possible, False otherwise.
    """
    import os
    import logging
    import inspect
    try:
        import torch
        model = torch.jit.load(path, map_location='cpu')
        model.eval()
        model_type = kwargs.get('model_type', None)
        fname = os.path.basename(path).lower()
        # 1. Detect model type
        if not model_type:
            if 'bert' in fname:
                model_type = 'bert'
            elif 'gpt2' in fname:
                model_type = 'gpt2'
            elif 'vit' in fname or 'resnet' in fname:
                model_type = 'vision'
            elif 't5' in fname or 'bart' in fname:
                model_type = 'seq2seq'
            elif 'clip' in fname:
                model_type = 'clip'
            else:
                model_type = 'auto'
        # 2. Try input patterns by model type
        if model_type in ('bert', 'roberta', 'distilbert'):
            dummy_input_ids = torch.randint(0, 1000, (1, 8), dtype=torch.long)
            dummy_attention_mask = torch.ones((1, 8), dtype=torch.long)
            try:
                _ = model(dummy_input_ids, dummy_attention_mask)
                return True
            except Exception:
                try:
                    _ = model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask)
                    return True
                except Exception as e:
                    logging.warning(f"TorchScript BERT-like inference failed: {e}")
        elif model_type == 'gpt2':
            dummy_input_ids = torch.randint(0, 1000, (1, 8), dtype=torch.long)
            try:
                _ = model(dummy_input_ids)
                return True
            except Exception:
                try:
                    _ = model(input_ids=dummy_input_ids)
                    return True
                except Exception as e:
                    logging.warning(f"TorchScript GPT-2 inference failed: {e}")
                    logging.warning("TorchScript dynamic check: GPT-2 and similar models are known to have limited support. For best results, use BERT, DistilBERT, or ResNet.")
        elif model_type == 'vision':
            dummy = torch.randn(1, 3, 224, 224)
            try:
                _ = model(dummy)
                return True
            except Exception as e:
                logging.warning(f"Vision model inference failed: {e}")
                logging.warning("TorchScript dynamic check: Vision models may require special handling. For best results, use BERT, DistilBERT, or ResNet.")
        elif model_type == 'seq2seq':
            dummy_input_ids = torch.randint(0, 1000, (1, 8), dtype=torch.long)
            dummy_decoder_input_ids = torch.randint(0, 1000, (1, 8), dtype=torch.long)
            dummy_attention_mask = torch.ones((1, 8), dtype=torch.long)
            try:
                _ = model(dummy_input_ids, dummy_decoder_input_ids, dummy_attention_mask)
                return True
            except Exception:
                try:
                    _ = model(input_ids=dummy_input_ids, decoder_input_ids=dummy_decoder_input_ids, attention_mask=dummy_attention_mask)
                    return True
                except Exception as e:
                    logging.warning(f"Seq2Seq model inference failed: {e}")
                    logging.warning("TorchScript dynamic check: T5/BART and similar models are known to have limited support. For best results, use BERT, DistilBERT, or ResNet.")
        elif model_type == 'clip':
            dummy_input_ids = torch.randint(0, 1000, (1, 8), dtype=torch.long)
            dummy_pixel_values = torch.randn(1, 3, 224, 224)
            try:
                _ = model(dummy_input_ids, dummy_pixel_values)
                return True
            except Exception:
                try:
                    _ = model(input_ids=dummy_input_ids, pixel_values=dummy_pixel_values)
                    return True
                except Exception as e:
                    logging.warning(f"CLIP model inference failed: {e}")
                    logging.warning("TorchScript dynamic check: CLIP and similar models are known to have limited support. For best results, use BERT, DistilBERT, or ResNet.")
        # 3. Fallback: try generic patterns
        try:
            dummy = torch.zeros(1, 8)
            _ = model(dummy)
            return True
        except Exception:
            pass
        try:
            dummy = torch.zeros(1, 3, 224, 224)
            _ = model(dummy)
            return True
        except Exception:
            pass
        # 4. Introspect signature and try to match
        try:
            sig = inspect.signature(model.forward)
            args = []
            for name, param in sig.parameters.items():
                if 'input' in name:
                    args.append(torch.randint(0, 1000, (1, 8), dtype=torch.long))
                elif 'mask' in name:
                    args.append(torch.ones((1, 8), dtype=torch.long))
                elif 'pixel' in name:
                    args.append(torch.randn(1, 3, 224, 224))
                elif 'decoder' in name:
                    args.append(torch.randint(0, 1000, (1, 8), dtype=torch.long))
                else:
                    args.append(torch.zeros(1, 8))
            _ = model(*args)
            return True
        except Exception as e:
            logging.warning(f"Signature introspection inference failed: {e}")
        # User-friendly message for unsupported models
        unsupported = ["gpt2", "llama", "mistral", "qwen", "mixtral", "vit", "t5", "bart", "clip"]
        if any(u in fname for u in unsupported):
            logging.warning(f"TorchScript dynamic check: Model '{fname}' is known to have limited or unstable TorchScript support. For best results, use BERT, DistilBERT, or ResNet.")
        return False
    except Exception as e:
        logging.warning(f"TorchScript dynamic check error: {e}")
        return False 