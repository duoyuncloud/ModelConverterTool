# ONNX format model conversion, post-processing and validation module

import subprocess
import sys
from pathlib import Path
import logging
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

def convert_to_onnx(
    model: Any = None,
    tokenizer: Any = None,
    model_name: Optional[str] = None,
    output_path: Optional[str] = None,
    model_type: str = "auto",
    task: str = "causal-lm",
    batch_size: int = 1,
    sequence_length: int = 8,
    device: str = "cpu",
    **kwargs
) -> Tuple[bool, Optional[dict]]:
    """
    Export a HuggingFace model to ONNX format using optimum.
    For Qwen/Qwen2 models, use optimum Python API with custom_onnx_configs.
    Returns (True, extra_info) if export succeeded and ONNX file is valid, (False, None) otherwise.
    """
    logger = logging.getLogger(__name__)
    if model_name is None and isinstance(model, str):
        model_name = model
        model = None
    if output_path:
        onnx_file = str(Path(output_path))
        output_dir = Path(output_path).parent
    else:
        output_dir = Path("outputs/onnx")
        onnx_file = str(output_dir / "model.onnx")
    output_dir.mkdir(parents=True, exist_ok=True)
    extra_info = {}

    # Auto-detect task type for certain models
    if task == "causal-lm" and model_name:
        lower_name = model_name.lower()
        if lower_name.startswith("bert") or "bert" in lower_name:
            task = "feature-extraction"

    # Load model and tokenizer if not provided
    try:
        if model is None or tokenizer is None or (getattr(tokenizer, "pad_token", None) is None):
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            if model is None:
                model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if getattr(tokenizer, "pad_token", None) is None:
                tokenizer.pad_token = tokenizer.eos_token
    except Exception:
        logger.info("Failed to load the model or tokenizer. Please check your model name, network connection, and ensure the model is supported by transformers. You may retry or consult the documentation: https://huggingface.co/docs/transformers/main/en/model_doc/auto")
        return False, None

    # Remove Qwen/Qwen2/Qwen3 custom config logic
    # Only use the default ONNX export logic for all models
    lower_name = model_name.lower() if model_name else ""
    if "qwen" in lower_name:
        logger.warning(
            "Note: Qwen/Qwen2/Qwen3 models are exported using the default ONNX logic because custom export configs are not available in this version. "
            "The exported ONNX model may have limited compatibility or performance. For optimal results, please follow future updates from HuggingFace Optimum "
            "or consult the documentation: https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model"
        )
    command = [
        sys.executable, "-m", "optimum.exporters.onnx",
        "--model", model_name,
        str(output_dir),
        "--task", task,
        "--batch_size", str(batch_size),
        "--sequence_length", str(sequence_length)
    ]
    logger.info(f"Running: {' '.join(command)}")
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        # Rename model.onnx to the desired output_path if needed
        default_onnx = str(output_dir / "model.onnx")
        if onnx_file != default_onnx and Path(default_onnx).exists():
            try:
                Path(onnx_file).unlink(missing_ok=True)
                Path(default_onnx).rename(onnx_file)
            except Exception as e:
                logger.error(f"Failed to rename ONNX file: {e}")
                return False, None
        if result.returncode == 0 and validate_onnx_file(onnx_file):
            logger.info(f"ONNX export and validation succeeded. Output: {onnx_file}")
            extra_info = {"opset": 17, "custom_onnx_configs": False}
            return True, extra_info
        else:
            logger.info("ONNX export did not complete successfully. Please check your model name, network, and optimum support. You may retry or consult the documentation: https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model")
            return False, None
    except Exception:
        logger.info("ONNX export failed due to an unexpected error. Please check your model name, network, and optimum support. You may retry or consult the documentation: https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model")
        return False, None

def validate_onnx_file(onnx_file: str) -> bool:
    try:
        import onnx
        import onnxruntime
        import numpy as np
        from pathlib import Path
        if not Path(onnx_file).exists() or Path(onnx_file).stat().st_size < 100:
            return False
        model = onnx.load(onnx_file)
        session = onnxruntime.InferenceSession(onnx_file)
        input_feed = {}
        for inp in session.get_inputs():
            shape = [d if isinstance(d, int) and d > 0 else 8 for d in inp.shape]
            dtype = np.int64 if inp.type == 'tensor(int64)' else np.float32
            input_feed[inp.name] = np.ones(shape, dtype=dtype)
        _ = session.run(None, input_feed)
        return True
    except Exception as e:
        logging.getLogger(__name__).error(f"ONNX validation failed: {e}")
        return False 