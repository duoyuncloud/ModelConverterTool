import logging
from pathlib import Path
from typing import Any, Optional
from model_converter_tool.utils import auto_load_model_and_tokenizer, get_calibration_dataset

logger = logging.getLogger(__name__)


def convert_to_gptq(
    model: Any,
    tokenizer: Any,
    model_name: str,
    output_path: str,
    model_type: str,
    device: str,
    quantization: Optional[str] = None,
    use_large_calibration: bool = False,
    quantization_config: dict = None,
) -> tuple:
    """
    Export model to GPTQ quantization format.
    """
    try:
        model, tokenizer = auto_load_model_and_tokenizer(model, tokenizer, model_name, model_type)
        import re
        from gptqmodel import GPTQModel, QuantizeConfig

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build quantize config
        bits = 4
        group_size = 128
        if quantization_config:
            allowed_keys = {
                "bits",
                "dynamic",
                "group_size",
                "damp_percent",
                "damp_auto_increment",
                "desc_act",
                "static_groups",
                "sym",
                "true_sequential",
                "lm_head",
                "quant_method",
                "format",
                "mse",
                "parallel_packing",
                "meta",
                "device",
                "pack_dtype",
                "adapter",
                "rotation",
                "is_marlin_format",
                "desc",
            }
            filtered_config = {k: v for k, v in quantization_config.items() if k in allowed_keys}
            quantize_config = QuantizeConfig(**filtered_config)
        elif quantization:
            m = re.match(r"(\d+)bit-(\d+)g", quantization)
            if m:
                bits = int(m.group(1))
                group_size = int(m.group(2))
            quantize_config = QuantizeConfig(bits=bits, group_size=group_size)
        else:
            quantize_config = QuantizeConfig(bits=bits, group_size=group_size)

        # Calibration + quantization
        calibration_dataset = get_calibration_dataset(use_large_calibration, tag="GPTQ")
        model = GPTQModel.from_pretrained(
            model_name, quantize_config, device=(device if device in ["cuda", "mps"] else "cpu")
        )
        model.quantize(calibration_dataset)
        model.save_pretrained(str(output_dir))

        logger.info(f"GPTQ quantization completed: {output_dir}")
        return True, None
    except Exception as e:
        logger.error(f"GPTQ conversion error: {e}")
        return False, None


def validate_gptq_file(path: str, *args, **kwargs) -> bool:
    """
    Static validation for GPTQ files. Accepts either a file or directory path.
    If a directory is given, uses it directly. If a file is given, uses its parent directory.
    Returns True if the model can be loaded, False otherwise.
    Prints detailed exception info on failure for debugging.
    """
    from pathlib import Path

    p = Path(path)
    if p.is_file():
        path = str(p.parent)
    elif p.is_dir():
        path = str(p)
    else:
        print(f"[validate_gptq_file] Path does not exist: {path}")
        return False
    try:
        from gptqmodel import GPTQModel

        _ = GPTQModel.load(path)
        print("[validate_gptq_file] Loaded GPTQ model successfully.")
        return True
    except ImportError:
        print("[validate_gptq_file] gptqmodel not installed.")
        return False
    except Exception as e:
        import traceback

        print(f"[validate_gptq_file] Exception: {e}\n" + traceback.format_exc())
        return False


def can_infer_gptq_file(path: str, *args, **kwargs) -> bool:
    """
    Dynamic check for GPTQ files.
    Load the model and run a dummy inference to verify usability.
    Returns True if inference succeeds, False otherwise.
    """
    import traceback

    try:
        from gptqmodel import GPTQModel
        import torch

        # 1. Load model
        try:
            model = GPTQModel.load(path)
            logger.info(f"[GPTQ] Model loaded from {path}")
        except Exception as e:
            logger.error(f"[GPTQ] Failed to load model: {e}\n{traceback.format_exc()}")
            return False

        # 2. Detect architecture
        arch = getattr(model, "arch", None)
        if arch is None and hasattr(model, "config"):
            arch_list = getattr(model.config, "architectures", [])
            arch = arch_list[0] if arch_list else None
        logger.info(f"[GPTQ] Detected architecture: {arch}")

        # 3. Get tokenizer if available
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None and hasattr(model, "get_tokenizer"):
            try:
                tokenizer = model.get_tokenizer()
                logger.info("[GPTQ] Tokenizer loaded via get_tokenizer()")
            except Exception as e:
                logger.warning(f"[GPTQ] Failed to get tokenizer: {e}")
                tokenizer = None

        # 4. Determine device of model parameters (default to cpu)
        model_device = None
        try:
            for p in model.parameters():
                model_device = p.device
                break
        except Exception:
            model_device = None
        if model_device is None:
            try:
                for p in getattr(model, "model", model).parameters():
                    model_device = p.device
                    break
            except Exception:
                model_device = None
        if model_device is None:
            model_device = torch.device("cpu")
        logger.info(f"[GPTQ] Using device: {model_device}")

        # 5. Prepare prompt
        prompt = "Hello world!"

        # 6. Define inference helper
        def generate_output(input_data):
            if isinstance(input_data, str):
                return model.generate(input_data)[0]
            else:
                input_ids = input_data.to(model_device)
                return model.generate(input_ids)[0]

        # 7. Run inference with fallback logic
        try:
            if tokenizer:
                # Try tokenized input first
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                output = generate_output(input_ids)
                _ = tokenizer.decode(output)
            else:
                # No tokenizer, try string input directly
                output = generate_output(prompt)
            logger.info("[GPTQ] Inference test succeeded.")
            return True
        except Exception as e:
            logger.warning(f"[GPTQ] Inference with tokenizer or string input failed: {e}")

            # If tokenizer present, try string input fallback
            if tokenizer:
                try:
                    output = generate_output(prompt)
                    _ = tokenizer.decode(output)
                    logger.info("[GPTQ] Inference fallback with string input succeeded.")
                    return True
                except Exception as e2:
                    logger.error(f"[GPTQ] Inference fallback also failed: {e2}\n{traceback.format_exc()}")
                    return False
            else:
                return False

    except ImportError:
        logger.error("[GPTQ] gptqmodel not installed.")
        return False
    except Exception as e:
        logger.error(f"[GPTQ] Unexpected error: {e}\n{traceback.format_exc()}")
        return False
