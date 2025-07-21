# ONNX format model conversion, post-processing and validation module

import subprocess
import sys
from pathlib import Path
import logging
from typing import Any, Optional, Tuple
from model_converter_tool.utils import auto_load_model_and_tokenizer
import os

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
    Always outputs to a directory, with the ONNX file named 'model.onnx'.
    """
    if model_name is None and isinstance(model, str):
        model_name = model
        model = None
    # Always treat output_path as a directory
    if output_path:
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        onnx_file = str(output_dir / "model.onnx")
    else:
        output_dir = Path("outputs/onnx")
        output_dir.mkdir(parents=True, exist_ok=True)
        onnx_file = str(output_dir / "model.onnx")
    extra_info = {}
    # Auto-detect task type for certain models
    if task == "causal-lm" and model_name:
        lower_name = model_name.lower()
        if lower_name.startswith("bert") or "bert" in lower_name:
            task = "feature-extraction"
    # Robust model/tokenizer auto-loading
    model, tokenizer = auto_load_model_and_tokenizer(model, tokenizer, model_name, model_type)
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
        default_onnx = str(output_dir / "model.onnx")
        # Always validate the ONNX file, not the directory
        if result.returncode == 0 and validate_onnx_file(default_onnx):
            logger.info(f"ONNX export and validation succeeded. Output: {default_onnx}")
            extra_info = {"opset": 17, "custom_onnx_configs": False}
            return True, extra_info
        else:
            logger.info("ONNX export did not complete successfully. Please check your model name, network, and optimum support. You may retry or consult the documentation: https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model")
            return False, None
    except Exception:
        logger.info("ONNX export failed due to an unexpected error. Please check your model name, network, and optimum support. You may retry or consult the documentation: https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model")
        return False, None

def validate_onnx_file(path: str, *args, **kwargs) -> bool:
    """
    Static validation for ONNX files. Accepts either a file or directory path.
    If a directory is given, looks for 'model.onnx' inside.
    Returns True if the file passes static validation, False otherwise.
    Now prints detailed exception info on failure.
    """
    import os
    if os.path.isdir(path):
        # If a directory is given, look for model.onnx inside
        candidate = os.path.join(path, "model.onnx")
        if os.path.exists(candidate):
            path = candidate
        else:
            print(f"[validate_onnx_file] Directory given but model.onnx not found: {path}")
            return False
    if not os.path.exists(path):
        print(f"[validate_onnx_file] File does not exist: {path}")
        return False
    try:
        import onnx
        model = onnx.load(path)
        onnx.checker.check_model(model)
        return True
    except ImportError:
        print("[validate_onnx_file] onnx not installed.")
        return False
    except Exception as e:
        import traceback
        print(f"[validate_onnx_file] Exception: {e}\n" + traceback.format_exc())
        return False

def can_infer_onnx_file(path: str, *args, **kwargs) -> bool:
    """
    Dynamic check for ONNX files. Loads the file with onnxruntime and runs a real dummy inference.
    Returns True if inference is possible, False otherwise. Prints/logs the actual exception if it fails.
    """
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(path)
        inputs = sess.get_inputs()
        if not inputs:
            print("[ONNX check] No inputs found in the model.")
            return False
        dummy = {}
        for inp in inputs:
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
            # Determine dtype from input type string
            dtype_str = inp.type
            if dtype_str == 'tensor(int64)':
                dtype = np.int64
            elif dtype_str == 'tensor(float)' or dtype_str == 'tensor(float32)':
                dtype = np.float32
            elif dtype_str == 'tensor(int32)':
                dtype = np.int32
            elif dtype_str == 'tensor(float16)':
                dtype = np.float16
            else:
                print(f"[ONNX check] Unknown input dtype {dtype_str}, defaulting to float32.")
                dtype = np.float32
            print(f"[ONNX check] Creating dummy input for '{inp.name}' with dtype {dtype} and shape {shape}")
            dummy[inp.name] = np.zeros(shape, dtype=dtype)
        _ = sess.run(None, dummy)
        return True
    except ImportError:
        print("[ONNX check] onnxruntime not installed.")
        return False
    except Exception as e:
        print(f"[ONNX check] Exception during inference: {e}")
        return False 