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
    # Support both old and new signatures for backward compatibility
    # If only model_name is given, treat as old signature
    if model_name is None and isinstance(model, str):
        model_name = model
        model = None
    # Determine output_dir and onnx_file
    if output_path and output_path.endswith(".onnx"):
        onnx_file = str(Path(output_path))
        output_dir = Path(output_path).parent
    else:
        output_dir = Path(output_path) if output_path else Path("outputs/onnx")
        onnx_file = str(output_dir / "model.onnx")
    output_dir.mkdir(parents=True, exist_ok=True)
    extra_info = {}

    # Detect Qwen/Qwen2 model
    is_qwen = model_name and ("qwen2" in model_name.lower() or "qwen" in model_name.lower())
    if is_qwen:
        try:
            logger.info("Detected Qwen/Qwen2 model, using optimum Python API with custom_onnx_configs.")
            from optimum.exporters.onnx import export
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            import importlib.util
            # Load custom Qwen2 OnnxConfig
            qwen_config_path = Path(__file__).parent.parent.parent / "qwen2_onnx_config.py"
            spec = importlib.util.spec_from_file_location("qwen2_onnx_config", str(qwen_config_path))
            qwen2_onnx_config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(qwen2_onnx_config)
            # Load model and tokenizer if not provided
            if model is None:
                model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            # Ensure pad_token exists
            if getattr(tokenizer, "pad_token", None) is None:
                tokenizer.pad_token = tokenizer.eos_token
            # Prepare custom_onnx_configs
            custom_onnx_configs = {"causal-lm": qwen2_onnx_config.get_config(model_name, task="causal-lm")}
            # Export to ONNX
            export(
                model=model,
                config=custom_onnx_configs["causal-lm"],
                opset=17,
                output=Path(onnx_file),
                device=device,
            )
            if validate_onnx_file(onnx_file):
                logger.info("ONNX export and validation succeeded for Qwen/Qwen2.")
                extra_info = {"opset": 17, "custom_onnx_configs": True}
                return True, extra_info
            else:
                logger.error("ONNX export or validation failed for Qwen/Qwen2.")
                return False, None
        except Exception as e:
            logger.error(f"Qwen/Qwen2 ONNX export failed: {e}")
            return False, None
    # Default: use CLI for other models
    command = [
        sys.executable, "-m", "optimum.exporters.onnx",
        "--model", model_name,
        str(output_dir),
        "--task", task,
        "--batch_size", str(batch_size),
        "--sequence_length", str(sequence_length)
    ]
    logger.info(f"Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0 and validate_onnx_file(onnx_file):
        logger.info("ONNX export and validation succeeded.")
        extra_info = {"opset": 17, "custom_onnx_configs": False}
        return True, extra_info
    else:
        logger.error(f"ONNX export or validation failed.\nStdout: {result.stdout}\nStderr: {result.stderr}")
        # Friendly error for Qwen/Qwen2
        if (
            "custom or unsupported architecture" in result.stderr or
            "custom_onnx_configs" in result.stderr or
            (model_name and ("qwen2" in model_name.lower() or "qwen" in model_name.lower()))
        ):
            logger.error(
                "Qwen/Qwen2 models require Python API + custom_onnx_configs export. See: https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model#custom-export-of-transformers-models"
            )
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