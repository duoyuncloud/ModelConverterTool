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
    **kwargs,
) -> Tuple[bool, Optional[dict]]:
    """
    Export model to ONNX format.

    Args:
        model: Loaded model object (optional)
        tokenizer: Loaded tokenizer object (optional)
        model_name: Source model name or path
        output_path: Output directory path (ONNX will always be saved as model.onnx inside this directory)
        model_type: Model type
        task: Task type for export
        batch_size: Batch size for dummy input
        sequence_length: Sequence length for dummy input
        device: Device to use for export
        kwargs: Additional arguments

    Returns:
        Tuple (success: bool, extra_info: dict or None)
        success: True if export succeeded, False otherwise
        extra_info: None if successful, or a dict with error details if failed
    """
    if model_name is None and isinstance(model, str):
        model_name = model
        model = None
    # Always treat output_path as a directory
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_file = output_dir / "model.onnx"
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

            if model is None:
                model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if getattr(tokenizer, "pad_token", None) is None:
                tokenizer.pad_token = tokenizer.eos_token
    except Exception:
        logger.info(
            "Failed to load the model or tokenizer. Please check your model name, network connection, and ensure the model is supported by transformers. You may retry or consult the documentation: https://huggingface.co/docs/transformers/main/en/model_doc/auto"
        )
        return False, None
    lower_name = model_name.lower() if model_name else ""
    if "qwen" in lower_name:
        logger.warning(
            "Note: Qwen/Qwen2/Qwen3 models are exported using the default ONNX logic because custom export configs are not available in this version. "
            "The exported ONNX model may have limited compatibility or performance. For optimal results, please follow future updates from HuggingFace Optimum "
            "or consult the documentation: https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model"
        )
    command = [
        sys.executable,
        "-m",
        "optimum.exporters.onnx",
        "--model",
        model_name,
        str(output_dir),
        "--task",
        task,
        "--batch_size",
        str(batch_size),
        "--sequence_length",
        str(sequence_length),
    ]
    logger.info(f"Running: {' '.join(command)}")
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        # Always validate the ONNX file, not the directory
        if result.returncode == 0 and validate_onnx_file(onnx_file):
            logger.info(f"ONNX export and validation succeeded. Output: {onnx_file}")
            extra_info = {"opset": 17, "custom_onnx_configs": False}
            return True, extra_info
        else:
            logger.info(
                "ONNX export did not complete successfully. Please check your model name, network, and optimum support. You may retry or consult the documentation: https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model"
            )
            return False, None
    except Exception:
        logger.info(
            "ONNX export failed due to an unexpected error. Please check your model name, network, and optimum support. You may retry or consult the documentation: https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model"
        )
        return False, None


def validate_onnx_file(path: str, *args, **kwargs) -> bool:
    """
    Static validation for ONNX files. Accepts either a file or directory path.
    If a directory is given, looks for 'model.onnx' inside.
    Returns True if the file passes static validation, False otherwise.
    Prints detailed exception info on failure for debugging.
    """
    from pathlib import Path

    p = Path(path)
    if p.is_dir():
        candidate = p / "model.onnx"
        if candidate.exists():
            path = str(candidate)
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


def can_infer_onnx_file(path: str, *args, **kwargs):
    """
    Dynamic check for ONNX files. Loads the model and runs a real dummy inference.
    Returns (True, None) if inference is possible, (False, error_message) otherwise.
    """
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(path)
        input_name = sess.get_inputs()[0].name
        dummy_input = np.zeros(sess.get_inputs()[0].shape, dtype=np.float32)
        _ = sess.run(None, {input_name: dummy_input})
        return True, None
    except Exception as e:
        return False, str(e)
