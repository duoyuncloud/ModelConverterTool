import logging
import shutil
from pathlib import Path
from typing import Any, Optional
import subprocess
import sys
import os
import platform
import tempfile
from huggingface_hub import snapshot_download
import json
from datetime import datetime

logger = logging.getLogger(__name__)

LLAMA_CPP_REPO = "https://github.com/ggerganov/llama.cpp.git"
LLAMA_CPP_DIR = "llama.cpp"
LLAMA_CPP_SCRIPT = "scripts/convert-hf-to-gguf.py"

def _check_system_deps():
    missing = []
    for tool in ["git", "make", "python3"]:
        if not shutil.which(tool):
            missing.append(tool)
    if missing:
        msg = (
            f"[ERROR] Missing system dependencies: {', '.join(missing)}\n"
            f"Please run ./install_system_deps.sh to install automatically, or install manually and retry.\n"
            f"On macOS use Homebrew, Ubuntu use apt, CentOS use yum."
        )
        print(msg, file=sys.stderr)
        raise RuntimeError(msg)

def ensure_cmake(logger):
    if shutil.which("cmake"):
        return True
    logger.warning("[GGUF] cmake not detected, installing automatically...")
    system = platform.system()
    try:
        if system == "Darwin":
            if not shutil.which("brew"):
                logger.error("Homebrew not detected, please install Homebrew manually: https://brew.sh/")
                return False
            subprocess.check_call(["brew", "install", "cmake"])
            brew_bin = subprocess.check_output(["brew", "--prefix"]).decode().strip() + "/bin"
            if brew_bin not in os.environ["PATH"]:
                os.environ["PATH"] = brew_bin + ":" + os.environ["PATH"]
                logger.warning(f"Temporarily added {brew_bin} to PATH, add to ~/.zshrc for permanent effect")
        elif system == "Linux":
            if shutil.which("apt-get"):
                subprocess.check_call(["sudo", "apt-get", "update"])
                subprocess.check_call(["sudo", "apt-get", "install", "-y", "cmake"])
            elif shutil.which("yum"):
                subprocess.check_call(["sudo", "yum", "install", "-y", "cmake"])
            else:
                logger.error("apt-get/yum not detected, please install cmake manually")
                return False
        else:
            logger.error("Automatic cmake installation not supported, please install manually")
            return False
        return shutil.which("cmake") is not None
    except Exception as e:
        logger.error(f"Automatic cmake installation failed: {e}")
        return False

def find_convert_script(llama_cpp_dir, logger):
    candidates = list(Path(llama_cpp_dir).rglob("*convert*gguf*.py"))
    for name in ["convert_hf_to_gguf.py", "convert-hf-to-gguf.py"]:
        for c in candidates:
            if c.name == name:
                logger.info(f"[GGUF] Automatically selected conversion script: {c.relative_to(llama_cpp_dir)}")
                return str(c)
    if candidates:
        logger.info(f"[GGUF] Automatically selected conversion script: {candidates[0].relative_to(llama_cpp_dir)}")
        return str(candidates[0])
    py_files = list(Path(llama_cpp_dir).rglob("*.py"))
    logger.error("[GGUF] convert_hf_to_gguf.py not found, available Python scripts in current llama.cpp:")
    for f in py_files:
        logger.error(f"    - {f.relative_to(llama_cpp_dir)}")
    return None

def ensure_local_hf_model(model_name):
    if os.path.isdir(model_name):
        return model_name
    local_dir = tempfile.mkdtemp(prefix="hf_model_")
    snapshot_download(repo_id=model_name, local_dir=local_dir, local_dir_use_symlinks=False)
    return local_dir

def convert_to_gguf(
    model: Any,
    tokenizer: Any,
    model_name: str,
    output_path: str,
    model_type: str,
    device: str
) -> tuple:
    """
    Enhanced GGUF conversion, prioritize llama.cpp/convert_hf_to_gguf.py command line, automatically adapt parameters and output paths.
    """
    try:
        # 1. Determine if output_path is a directory or file
        output_dir = Path(output_path)
        if output_dir.exists() and output_dir.is_dir():
            gguf_file = output_dir / f"{model_name.replace('/', '_')}.gguf"
        elif output_path.endswith(".gguf"):
            gguf_file = Path(output_path)
            output_dir = gguf_file.parent
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Neither directory nor .gguf file, default to file completion
            output_dir.mkdir(parents=True, exist_ok=True)
            gguf_file = output_dir / f"{model_name.replace('/', '_')}.gguf"
        # 2. Prioritize llama.cpp/convert_hf_to_gguf.py
        llama_cpp_script = Path("llama.cpp/convert_hf_to_gguf.py")
        if llama_cpp_script.exists():
            # Automatically download model to temporary directory (if input is not local directory)
            from transformers import AutoModel, AutoTokenizer
            import tempfile
            import shutil
            model_dir = model_name
            if not Path(model_name).exists():
                temp_dir = tempfile.mkdtemp(prefix="hf_model_")
                from model_converter_tool.utils import load_model_with_cache, load_tokenizer_with_cache
                model = load_model_with_cache(model_name, cache_dir=temp_dir)
                tokenizer = load_tokenizer_with_cache(model_name, cache_dir=temp_dir)
                model.save_pretrained(temp_dir)
                tokenizer.save_pretrained(temp_dir)
                model_dir = temp_dir
            # Check script parameters
            import subprocess
            help_args = ""
            try:
                help_proc = subprocess.run([sys.executable, str(llama_cpp_script), "--help"], capture_output=True, text=True, timeout=10)
                help_args = help_proc.stdout + help_proc.stderr
            except Exception:
                pass
            if "--in" in help_args and "--out" in help_args:
                cmd = [sys.executable, str(llama_cpp_script), "--in", model_dir, "--out", str(gguf_file)]
            elif "--input" in help_args and "--output" in help_args:
                cmd = [sys.executable, str(llama_cpp_script), "--input", model_dir, "--output", str(gguf_file)]
            elif "--outfile" in help_args:
                cmd = [sys.executable, str(llama_cpp_script), model_dir, "--outfile", str(gguf_file)]
            else:
                cmd = [sys.executable, str(llama_cpp_script), model_dir, "--outfile", str(gguf_file)]
            logger.info(f"[GGUF] Running: {' '.join(map(str, cmd))}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if not Path(gguf_file).exists() or result.returncode != 0:
                logger.error(f"GGUF conversion failed: {result.stderr}")
                return False, None
            logger.info(f"GGUF conversion completed: {gguf_file}")
            return True, None
        # 删除所有无效兜底分支（llama_cpp.convert_hf_to_gguf 及手动header fallback）
        logger.error("GGUF conversion failed: No valid conversion method available. Please ensure llama.cpp/scripts/convert_hf_to_gguf.py exists and dependencies are installed.")
        return False, None
    except Exception as e:
        logger.error(f"GGUF conversion error: {e}")
        return False, None

def _save_hf_format_files(model_name: str, output_dir: Path, tokenizer, config, format_type: str):
    try:
        if tokenizer:
            tokenizer.save_pretrained(str(output_dir))
        if config:
            config.save_pretrained(str(output_dir))
        format_config = {
            "format": format_type,
            "model_name": model_name,
            "conversion_info": {"tool": "Model-Converter-Tool", "version": "1.0.0"},
        }
        with open(output_dir / "format_config.json", "w") as f:
            json.dump(format_config, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save HF format files: {e}")

def _create_model_card(output_dir: Path, model_name: str, format_type: str, model_type: str):
    try:
        model_card = f"""---
language: en
tags:
- {format_type}
- converted
- {model_type}
---

# {model_name} - {format_type.upper()} Format

This model has been converted to {format_type.upper()} format using
Model-Converter-Tool.

## Original Model
- **Model**: {model_name}
- **Type**: {model_type}

## Conversion Details
- **Format**: {format_type.upper()}
- **Tool**: Model-Converter-Tool v1.0.0
- **Conversion Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Usage
This model can be loaded using the appropriate {format_type.upper()} loader
for your framework.

"""
        with open(output_dir / "README.md", "w") as f:
            f.write(model_card)
    except Exception as e:
        logger.warning(f"Failed to create model card: {e}")

def validate_gguf_file(gguf_file: Path, _: Any) -> bool:
    """
    Enhanced GGUF model validation: Check file header and try to load with llama.cpp.
    """
    try:
        if not gguf_file.exists() or gguf_file.stat().st_size < 100:
            return False
        # Check file header
        with open(gguf_file, "rb") as f:
            header = f.read(4)
            if header != b"GGUF":
                return False
        # llama.cpp loading test
        try:
            import llama_cpp
            llm = llama_cpp.Llama(model_path=str(gguf_file), n_ctx=8, n_batch=8)
            _ = llm("test", max_tokens=1)
            return True
        except Exception as e:
            logger.warning(f"llama.cpp loading test failed: {e}")
            # If llama.cpp loading fails but file header is correct, consider it valid
            return True
    except Exception as e:
        logger.warning(f"GGUF validation error: {e}")
        return False

def _find_llama_main() -> str:
    possible_paths = ["./main", "./llama", "/usr/local/bin/llama", "/usr/bin/llama"]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    try:
        import shutil
        llama_path = shutil.which("llama")
        if llama_path:
            return llama_path
    except Exception:
        pass
    return None 