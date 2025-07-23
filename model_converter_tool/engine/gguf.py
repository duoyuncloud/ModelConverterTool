import logging
from pathlib import Path
from typing import Any
import subprocess
import sys
import tempfile
import os
from model_converter_tool.utils import auto_load_model_and_tokenizer

logger = logging.getLogger(__name__)


def convert_to_gguf(
    model: Any,
    tokenizer: Any,
    model_name: str,
    output_path: str,
    model_type: str,
    device: str,
) -> tuple:
    """
    Export model to GGUF format (no quantization).

    Args:
        model: Loaded model object
        tokenizer: Loaded tokenizer object
        model_name: Source model name or path
        output_path: Output directory path (GGUF will be saved as model.gguf inside this directory)
        model_type: Model type
        device: Device (unused)
    Returns:
        (success: bool, extra_info: dict or None)
    """
    try:
        model, tokenizer = auto_load_model_and_tokenizer(model, tokenizer, model_name, model_type)
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        gguf_file = output_dir / "model.gguf"

        llama_cpp_script = Path("tools/llama.cpp/convert_hf_to_gguf.py")
        if not llama_cpp_script.exists():
            logger.error("GGUF conversion failed: llama.cpp/convert_hf_to_gguf.py not found.")
            return False, None

        model_dir = model_name
        if not Path(model_name).exists():
            temp_dir = tempfile.mkdtemp(prefix="hf_model_")
            model.save_pretrained(temp_dir)
            tokenizer.save_pretrained(temp_dir)
            model_dir = temp_dir

        cmd = [sys.executable, str(llama_cpp_script), model_dir, "--outfile", str(gguf_file)]
        logger.info(f"[GGUF] Running: {' '.join(map(str, cmd))}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if not gguf_file.exists() or result.returncode != 0:
            logger.error(f"GGUF conversion failed: {result.stderr}")
            return False, None

        if not validate_gguf_file(gguf_file):
            logger.error(f"GGUF validation failed for file: {gguf_file}")
            return False, None

        logger.info(f"GGUF conversion completed: {gguf_file}")
        return True, None

    except Exception as e:
        logger.error(f"GGUF conversion error: {e}")
        return False, None


def validate_gguf_file(path: str) -> bool:
    p = Path(path)
    if p.is_dir():
        candidate = p / "model.gguf"
        if candidate.exists():
            path = str(candidate)
        else:
            print(f"[validate_gguf_file] Directory given but model.gguf not found: {path}")
            return False
    if not os.path.exists(path):
        print(f"[validate_gguf_file] File does not exist: {path}")
        return False
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                print(f"[validate_gguf_file] Invalid GGUF file magic: {magic}")
                return False
        return True
    except Exception as e:
        import traceback

        print(f"[validate_gguf_file] Exception: {e}\n" + traceback.format_exc())
        return False


def can_infer_gguf_file(path: Path) -> bool:
    try:
        import llama_cpp

        llm = llama_cpp.Llama(model_path=str(path), n_ctx=8, n_batch=8)
        _ = llm("Hello", max_tokens=1)
        return True
    except Exception:
        return False
