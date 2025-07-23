import logging
from pathlib import Path
from typing import Any
import subprocess
import sys
import tempfile
import json
from model_converter_tool.utils import auto_load_model_and_tokenizer
import os

logger = logging.getLogger(__name__)


def convert_to_gguf(
    model: Any,
    tokenizer: Any,
    model_name: str,
    output_path: str,
    model_type: str,
    device: str,
    quantization: str = None,
    use_large_calibration: bool = False,
    quantization_config: dict = None,
) -> tuple:
    """
    Export model to GGUF format.

    Args:
        model: Loaded model object
        tokenizer: Loaded tokenizer object
        model_name: Source model name or path
        output_path: Output directory path (GGUF will always be saved as model.gguf inside this directory)
        model_type: Model type
        device: Device
        quantization: Quantization string (optional)
        use_large_calibration: Unused for GGUF
        quantization_config: Dict with quantization parameters (optional)
    Returns:
        (success: bool, extra_info: dict or None)
    """
    try:
        # Robust model/tokenizer auto-loading
        model, tokenizer = auto_load_model_and_tokenizer(model, tokenizer, model_name, model_type)
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        gguf_file = output_dir / "model.gguf"
        llama_cpp_script = Path("tools/llama.cpp/convert_hf_to_gguf.py")
        if not llama_cpp_script.exists():
            logger.error(
                "GGUF conversion failed: llama.cpp/convert_hf_to_gguf.py not found. Please ensure it exists and dependencies are installed."
            )
            return False, None

        model_dir = model_name
        if not Path(model_name).exists():
            temp_dir = tempfile.mkdtemp(prefix="hf_model_")
            model.save_pretrained(temp_dir)
            tokenizer.save_pretrained(temp_dir)
            model_dir = temp_dir

        # Use --outfile to adapt to the updated script
        cmd = [sys.executable, str(llama_cpp_script), model_dir, "--outfile", str(gguf_file)]
        logger.info(f"[GGUF] Running: {' '.join(map(str, cmd))}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if not gguf_file.exists() or result.returncode != 0:
            logger.error(f"GGUF conversion failed: {result.stderr}")
            return False, None

        # Validate the GGUF file itself
        if not validate_gguf_file(gguf_file):
            logger.error(f"GGUF validation failed for file: {gguf_file}")
            return False, None

        logger.info(f"GGUF conversion completed: {gguf_file}")

        # Parse quantization config or infer defaults
        bits = quantization_config.get("bits") if quantization_config else None
        group_size = quantization_config.get("group_size") if quantization_config else None
        sym = quantization_config.get("sym") if quantization_config else None
        desc = quantization_config.get("desc") if quantization_config else None

        if bits is None and quantization:
            import re

            m = re.match(r"(\d+)bit", quantization)
            if m:
                bits = int(m.group(1))

        if group_size is None and quantization:
            import re

            m = re.match(r".*g(\d+)", quantization)
            if m:
                group_size = int(m.group(1))

        bits = bits if bits is not None else 4
        group_size = group_size if group_size is not None else 128
        sym = sym if sym is not None else False

        # Directly update config.json with quantization parameters
        config_path = gguf_file.parent / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                config.update(
                    {
                        "bits": bits,
                        "group_size": group_size,
                        "sym": sym,
                        "desc": desc,
                    }
                )
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2)
                logger.info(f"Quantization config patched into {config_path}")
            except Exception as e:
                logger.warning(f"Failed to patch quantization config: {e}")

        return True, None

    except Exception as e:
        logger.error(f"GGUF conversion error: {e}")
        return False, None


def validate_gguf_file(path: str, *args, **kwargs) -> bool:
    """
    Static validation for GGUF files. Accepts either a file or directory path.
    If a directory is given, looks for 'model.gguf' inside.
    Returns True if the file passes static validation, False otherwise.
    Prints detailed exception info on failure for debugging.
    """
    from pathlib import Path

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
        # Simple GGUF file header check
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


def can_infer_gguf_file(path: Path, *args, **kwargs) -> bool:
    """
    Dynamic check for GGUF files. Loads the file with llama_cpp and runs a real dummy inference.
    Returns True if inference is possible, False otherwise.
    """
    try:
        import llama_cpp

        llm = llama_cpp.Llama(model_path=str(path), n_ctx=8, n_batch=8)
        # Run a real dummy inference
        _ = llm("Hello", max_tokens=1)
        return True
    except ImportError:
        # llama_cpp not installed
        return False
    except Exception:
        return False
