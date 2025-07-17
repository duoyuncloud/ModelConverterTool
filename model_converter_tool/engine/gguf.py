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
    quantization_config: dict = None
) -> tuple:
    """
    Convert a model to GGUF format using llama.cpp/convert_hf_to_gguf.py as an external script.
    Args:
        model: Loaded model object
        tokenizer: Loaded tokenizer object
        model_name: Source model name or path
        output_path: Output file path or directory
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
        if output_dir.exists() and output_dir.is_dir():
            gguf_file = output_dir / f"{model_name.replace('/', '_')}.gguf"
        elif str(output_path).endswith(".gguf"):
            gguf_file = Path(output_path)
            output_dir = gguf_file.parent
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            gguf_file = output_dir / f"{model_name.replace('/', '_')}.gguf"
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
            if 'fake_weight' in locals() and fake_weight:
                from model_converter_tool.utils import patch_config_remove_quantization_config
                patch_config_remove_quantization_config(output_dir)
            return True, None
        logger.error("GGUF conversion failed: llama.cpp/convert_hf_to_gguf.py not found. Please ensure it exists and dependencies are installed.")
        return False, None
    except Exception as e:
        logger.error(f"GGUF conversion error: {e}")
        return False, None

def validate_gguf_file(path: Path, *args, **kwargs) -> bool:
    """
    Static validation for GGUF files. Checks if the file exists, has a valid GGUF header, and can be loaded by llama_cpp if available.
    Returns True if the file passes static validation, False otherwise.
    """
    if not os.path.exists(path):
        return False
    try:
        with open(path, 'rb') as f:
            header = f.read(4)
            if header != b'GGUF':
                return False
        try:
            import llama_cpp
            _ = llama_cpp.Llama(model_path=str(path), n_ctx=8, n_batch=8)
        except ImportError:
            # llama_cpp not installed, skip deep validation
            pass
        except Exception:
            return False
        return True
    except Exception:
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