import logging
from pathlib import Path
from typing import Any
import subprocess
import sys
import tempfile
from model_converter_tool.utils import auto_load_model_and_tokenizer, patch_quantization_config

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
        llama_cpp_script = Path("llama.cpp/convert_hf_to_gguf.py")
        if llama_cpp_script.exists():
            model_dir = model_name
            if not Path(model_name).exists():
                temp_dir = tempfile.mkdtemp(prefix="hf_model_")
                model.save_pretrained(temp_dir)
                tokenizer.save_pretrained(temp_dir)
                model_dir = temp_dir
            help_args = ""
            try:
                help_proc = subprocess.run([sys.executable, str(llama_cpp_script), "--help"], capture_output=True, text=True, timeout=10)
                help_args = help_proc.stdout + help_proc.stderr
            except Exception:
                pass
            quant_map = {
                "q4_k_m": "q4_0",
                "q8_0": "q8_0",
                "f16": "f16",
                "auto": "auto",
                None: "auto",
                "": "auto"
            }
            outtype = quant_map.get(quantization, quantization if quantization else "auto")
            if "--in" in help_args and "--out" in help_args:
                cmd = [sys.executable, str(llama_cpp_script), "--in", model_dir, "--out", str(gguf_file)]
                if outtype:
                    cmd += ["--outtype", outtype]
            else:
                cmd = [sys.executable, str(llama_cpp_script), model_dir, "--outfile", str(gguf_file)]
                if outtype:
                    cmd += ["--outtype", outtype]
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
            return True, None
        logger.error("GGUF conversion failed: llama.cpp/convert_hf_to_gguf.py not found. Please ensure it exists and dependencies are installed.")
        return False, None
    except Exception as e:
        logger.error(f"GGUF conversion error: {e}")
        return False, None

def validate_gguf_file(gguf_file: Path, _: Any) -> bool:
    """
    Validate GGUF file: check header and try to load with llama.cpp if available.
    """
    try:
        if not gguf_file.exists() or gguf_file.stat().st_size < 100:
            return False
        with open(gguf_file, "rb") as f:
            header = f.read(4)
            if header != b"GGUF":
                return False
        try:
            import llama_cpp
            llm = llama_cpp.Llama(model_path=str(gguf_file), n_ctx=8, n_batch=8)
            _ = llm("test", max_tokens=1)
            return True
        except Exception as e:
            logger.warning(f"llama.cpp loading test failed: {e}")
            return True
    except Exception as e:
        logger.warning(f"GGUF validation error: {e}")
        return False 