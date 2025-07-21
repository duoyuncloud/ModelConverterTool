import logging
from pathlib import Path
from typing import Any
import subprocess
import sys
import tempfile
from model_converter_tool.utils import auto_load_model_and_tokenizer, patch_quantization_config
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
        if llama_cpp_script.exists():
            model_dir = model_name
            if not Path(model_name).exists():
                temp_dir = tempfile.mkdtemp(prefix="hf_model_")
                model.save_pretrained(temp_dir)
                tokenizer.save_pretrained(temp_dir)
                model_dir = temp_dir
            # 直接用 --outfile 参数适配新版脚本
            cmd = [sys.executable, str(llama_cpp_script), model_dir, "--outfile", str(gguf_file)]
            logger.info(f"[GGUF] Running: {' '.join(map(str, cmd))}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if not Path(gguf_file).exists() or result.returncode != 0:
                logger.error(f"GGUF conversion failed: {result.stderr}")
                return False, None
            # Always validate the GGUF file, not the directory
            if result.returncode == 0 and validate_gguf_file(gguf_file):
                logger.info(f"GGUF conversion completed: {gguf_file}")
                # Parse quantization config
                bits = quantization_config.get("bits") if quantization_config else None
                group_size = quantization_config.get("group_size") if quantization_config else None
                sym = quantization_config.get("sym") if quantization_config else None
                desc = quantization_config.get("desc") if quantization_config else None
                # Try to infer from quantization string if not provided
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
                if bits is None:
                    bits = 4
                if group_size is None:
                    group_size = 128
                if sym is None:
                    sym = False
                patch_quantization_config(gguf_file.parent / "config.json", bits, group_size, sym, desc)
                # After saving model, patch config if fake_weight
                # if "fake_weight" in locals() and fake_weight:
                #     from model_converter_tool.utils import patch_config_remove_quantization_config
                #
                #     patch_config_remove_quantization_config(output_dir)
                return True, None
            logger.error(
                "GGUF conversion failed: llama.cpp/convert_hf_to_gguf.py not found. Please ensure it exists and dependencies are installed."
            )
            return False, None
        logger.error(
            "GGUF conversion failed: llama.cpp/convert_hf_to_gguf.py not found. Please ensure it exists and dependencies are installed."
        )
        return False, None
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
        # Insert actual GGUF validation logic here, e.g., open and check file header
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
