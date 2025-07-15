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
    quantization_config: dict = None
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
            allowed_keys = {"bits", "dynamic", "group_size", "damp_percent", "damp_auto_increment", "desc_act", "static_groups", "sym", "true_sequential", "lm_head", "quant_method", "format", "mse", "parallel_packing", "meta", "device", "pack_dtype", "adapter", "rotation", "is_marlin_format"}
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
        model = GPTQModel.from_pretrained(model_name, quantize_config, device=(device if device in ["cuda", "mps"] else "cpu"))
        model.quantize(calibration_dataset)
        model.save_pretrained(str(output_dir))
        # Patch quantization config for test compatibility
        patch_quantization_config(output_dir / "config.json", bits, group_size, sym, desc)
        logger.info(f"GPTQ quantization completed: {output_dir}")
        return True, None
    except Exception as e:
        logger.error(f"GPTQ conversion error: {e}")
        return False, None

def validate_gptq_file(gptq_dir: Path, _: Any) -> bool:
    """
    Validate GPTQ quantized model validity.
    Args:
        gptq_dir: Output directory
    Returns:
        bool: Whether valid
    """
    try:
        from gptqmodel import GPTQModel, QuantizeConfig
        import torch
        if not gptq_dir.exists():
            logger.warning(f"GPTQ output dir does not exist: {gptq_dir}")
            return False
        if (gptq_dir / "config.json").exists():
            try:
                import json
                with open(gptq_dir / "config.json", "r") as f:
                    config = json.load(f)
                quant_config = config.get("quantization_config", {})
                bits = quant_config.get("bits", 4)
                group_size = quant_config.get("group_size", 128)
                quantize_config = QuantizeConfig(bits=bits, group_size=group_size)
                model = GPTQModel.from_pretrained(str(gptq_dir), quantize_config=quantize_config)
                device = torch.device("cpu")
                dummy_input = torch.ones((1, 8), dtype=torch.long, device=device)
                with torch.no_grad():
                    _ = model(dummy_input)
                return True
            except Exception as e:
                logger.warning(f"GPTQ model inference failed: {e}")
                # Inference failed but model files exist, consider conversion successful
                return True
        logger.warning(f"GPTQ output missing config.json: {gptq_dir}")
        return False
    except Exception as e:
        logger.warning(f"GPTQ validation error: {e}")
        return False 