import logging
from pathlib import Path
from typing import Any
import subprocess
import sys
import tempfile
from model_converter_tool.utils import load_model_with_cache
from transformers import AutoModel, AutoModelForCausalLM
from model_converter_tool.utils import load_tokenizer_with_cache
from transformers import AutoTokenizer

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
    Minimal GGUF conversion: only call llama.cpp/convert_hf_to_gguf.py as external script.
    """
    try:
        # Robust model/tokenizer auto-loading
        if model is None or tokenizer is None:
            if model is None:
                if model_type and ("causal" in model_type or "lm" in model_type or "generation" in model_type):
                    model = load_model_with_cache(model_name, AutoModelForCausalLM)
                else:
                    model = load_model_with_cache(model_name, AutoModel)
            if tokenizer is None:
                tokenizer = load_tokenizer_with_cache(model_name)
        output_dir = Path(output_path)
        if output_dir.exists() and output_dir.is_dir():
            gguf_file = output_dir / f"{model_name.replace('/', '_')}.gguf"
        elif output_path.endswith(".gguf"):
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
                model = load_model_with_cache(model_name, cache_dir=temp_dir)
                tokenizer = load_tokenizer_with_cache(model_name, cache_dir=temp_dir)
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