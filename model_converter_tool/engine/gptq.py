import logging
from pathlib import Path
from typing import Any, Optional
from model_converter_tool.utils import auto_load_model_and_tokenizer, get_calibration_dataset, patch_quantization_config

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
    Args:
        model: Loaded model object
        tokenizer: Loaded tokenizer object
        model_name: Source model name or path
        output_path: Output file path
        model_type: Model type
        device: Device
        quantization: Quantization parameters (optional)
        use_large_calibration: Whether to use a large calibration set
    Returns:
        (success: bool, extra_info: dict or None)
    """
    try:
        # Robust model/tokenizer auto-loading
        model, tokenizer = auto_load_model_and_tokenizer(model, tokenizer, model_name, model_type)
        import re
        from gptqmodel import GPTQModel, QuantizeConfig

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Parse quantization config
        bits = 4
        group_size = 128
        sym = False
        desc = None
        if quantization_config:
            bits = quantization_config.get("bits", bits)
            group_size = quantization_config.get("group_size", group_size)
            sym = quantization_config.get("sym", sym)
            desc = quantization_config.get("desc", desc)
            # Only pass supported keys to QuantizeConfig
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
        calibration_dataset = get_calibration_dataset(use_large_calibration, tag="GPTQ")
        model = GPTQModel.from_pretrained(
            model_name, quantize_config, device=(device if device in ["cuda", "mps"] else "cpu")
        )
        model.quantize(calibration_dataset)
        model.save_pretrained(str(output_dir))
        # Patch quantization config for test compatibility
        patch_quantization_config(output_dir / "config.json", bits, group_size, sym, desc)
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
    Dynamic check for GPTQ files. Loads the model with GPTQModel and runs a real dummy inference.
    Adapts logic for OPT, Llama, Mistral, and other architectures. Ensures device compatibility for inference.
    Returns True if inference is possible, False otherwise.
    """
    import traceback

    try:
        from gptqmodel import GPTQModel
        import torch

        try:
            model = GPTQModel.load(path)
        except Exception as e:
            logger.error(f"[GPTQ] Failed to load model: {e}\n{traceback.format_exc()}")
            return False
        # Try to detect architecture
        arch = getattr(model, "arch", None)
        if arch is None and hasattr(model, "config"):
            arch = getattr(model.config, "architectures", [None])[0]
        logger.info(f"[GPTQ] Detected architecture: {arch}")
        # Try to get tokenizer
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None and hasattr(model, "get_tokenizer"):
            try:
                tokenizer = model.get_tokenizer()
            except Exception:
                tokenizer = None
        # Detect model device (cpu, cuda, mps, etc)
        model_device = None
        try:
            # Try to get device from model parameters
            for param in model.parameters():
                model_device = param.device
                break
        except Exception:
            pass
        if model_device is None:
            # Fallback: try to get from model.model (for HuggingFace)
            try:
                for param in getattr(model, "model", model).parameters():
                    model_device = param.device
                    break
            except Exception:
                pass
        if model_device is None:
            # Default to cpu
            model_device = torch.device("cpu")
        logger.info(f"[GPTQ] Using device: {model_device}")
        prompt = "Hello world!"
        try:
            if arch is not None:
                arch_l = arch.lower()
                if "opt" in arch_l:
                    # OPT models: use default prompt and tokenizer
                    if tokenizer is not None:
                        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                        # Move input_ids to model device
                        input_ids = input_ids.to(model_device)
                        output = model.generate(input_ids)[0]
                        _ = tokenizer.decode(output)
                    else:
                        output = model.generate(prompt)[0]
                elif "llama" in arch_l or "mistral" in arch_l or "mixtral" in arch_l:
                    # Llama/Mistral: try both string and token input
                    if tokenizer is not None:
                        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                        input_ids = input_ids.to(model_device)
                        output = model.generate(input_ids)[0]
                        _ = tokenizer.decode(output)
                    else:
                        output = model.generate(prompt)[0]
                else:
                    # Fallback: try string prompt
                    output = model.generate(prompt)[0]
                    if tokenizer is not None:
                        _ = tokenizer.decode(output)
            else:
                # Unknown arch: try both string and token input
                try:
                    output = model.generate(prompt)[0]
                    if tokenizer is not None:
                        _ = tokenizer.decode(output)
                except Exception:
                    if tokenizer is not None:
                        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                        input_ids = input_ids.to(model_device)
                        output = model.generate(input_ids)[0]
                        _ = tokenizer.decode(output)
            return True
        except Exception as e:
            logger.error(f"[GPTQ] Inference failed: {e}\n{traceback.format_exc()}")
            return False
    except ImportError:
        logger.error("[GPTQ] gptqmodel not installed.")
        return False
    except Exception as e:
        logger.error(f"[GPTQ] Unexpected error: {e}\n{traceback.format_exc()}")
        return False
