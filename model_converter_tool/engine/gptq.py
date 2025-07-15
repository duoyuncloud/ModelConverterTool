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

def validate_gptq_file(gptq_dir, _):
    try:
        from gptqmodel import GPTQModel
        model = GPTQModel.load(str(gptq_dir))
        tokens = model.generate("Test prompt")[0]
        _ = model.tokenizer.decode(tokens)
        return True
    except Exception as e:
        import logging, traceback
        logging.getLogger(__name__).error(f"GPTQ validation failed: {e}")
        traceback.print_exc()
        return False 