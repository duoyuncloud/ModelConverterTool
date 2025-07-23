import logging
from pathlib import Path
from typing import Any, Optional
from model_converter_tool.utils import (
    auto_load_model_and_tokenizer,
    get_calibration_dataset,
    patch_quantization_config_file,
)

logger = logging.getLogger(__name__)


def convert_to_awq(
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
    Export model to AWQ quantization format.
    """
    try:
        model, tokenizer = auto_load_model_and_tokenizer(model, tokenizer, model_name, model_type)
        import re
        from gptqmodel import GPTQModel, QuantizeConfig

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        bits = 4
        group_size = 128
        sym = False
        desc_act = None

        if quantization_config:
            bits = quantization_config.get("bits", bits)
            group_size = quantization_config.get("group_size", group_size)
            sym = quantization_config.get("sym", sym)
            desc_act = quantization_config.get("desc_act", desc_act)
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

        calibration_dataset = get_calibration_dataset(use_large_calibration, tag="AWQ")
        model = GPTQModel.from_pretrained(
            model_name, quantize_config, device=(device if device in ["cuda", "mps"] else "cpu")
        )
        model.quantize(calibration_dataset)
        model.save_pretrained(str(output_dir))

        patch_quantization_config_file(output_dir / "config.json", bits, group_size, sym, desc_act)

        logger.info(f"AWQ quantization completed: {output_dir}")
        return True, None
    except Exception as e:
        logger.error(f"AWQ conversion error: {e}")
        return False, None


def validate_awq_file(path: str, *args, **kwargs) -> bool:
    """
    Validate AWQ files by attempting to load the model.
    """
    from pathlib import Path

    p = Path(path)
    if p.is_file():
        path = str(p.parent)
    elif p.is_dir():
        path = str(p)
    else:
        print(f"[validate_awq_file] Path does not exist: {path}")
        return False
    try:
        from gptqmodel import GPTQModel

        _ = GPTQModel.load(path)
        print("[validate_awq_file] Loaded AWQ model successfully.")
        return True
    except ImportError:
        print("[validate_awq_file] gptqmodel not installed.")
        return False
    except Exception as e:
        import traceback

        print(f"[validate_awq_file] Exception: {e}\n" + traceback.format_exc())
        return False


def can_infer_awq_file(path: str, *args, **kwargs) -> bool:
    """
    Perform dummy inference to test AWQ model usability.
    """
    import traceback

    try:
        from gptqmodel import GPTQModel
        import torch

        try:
            model = GPTQModel.load(path)
        except Exception as e:
            logger.error(f"[AWQ] Failed to load model: {e}\n{traceback.format_exc()}")
            return False

        arch = getattr(model, "arch", None)
        if arch is None and hasattr(model, "config"):
            arch = getattr(model.config, "architectures", [None])[0]
        logger.info(f"[AWQ] Detected architecture: {arch}")

        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None and hasattr(model, "get_tokenizer"):
            try:
                tokenizer = model.get_tokenizer()
            except Exception:
                tokenizer = None

        model_device = None
        try:
            for param in model.parameters():
                model_device = param.device
                break
        except Exception:
            pass
        if model_device is None:
            try:
                for param in getattr(model, "model", model).parameters():
                    model_device = param.device
                    break
            except Exception:
                pass
        if model_device is None:
            model_device = torch.device("cpu")
        logger.info(f"[AWQ] Using device: {model_device}")

        prompt = "Hello world!"
        try:
            if arch:
                arch_l = arch.lower()
                if any(x in arch_l for x in ["opt", "llama", "mistral", "mixtral"]):
                    if tokenizer:
                        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model_device)
                        output = model.generate(input_ids)[0]
                        _ = tokenizer.decode(output)
                    else:
                        output = model.generate(prompt)[0]
                else:
                    output = model.generate(prompt)[0]
                    if tokenizer:
                        _ = tokenizer.decode(output)
            else:
                try:
                    output = model.generate(prompt)[0]
                    if tokenizer:
                        _ = tokenizer.decode(output)
                except Exception:
                    if tokenizer:
                        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model_device)
                        output = model.generate(input_ids)[0]
                        _ = tokenizer.decode(output)
            return True
        except Exception as e:
            logger.error(f"[AWQ] Inference failed: {e}\n{traceback.format_exc()}")
            return False
    except ImportError:
        logger.error("[AWQ] gptqmodel not installed.")
        return False
    except Exception as e:
        logger.error(f"[AWQ] Unexpected error: {e}\n{traceback.format_exc()}")
        return False
