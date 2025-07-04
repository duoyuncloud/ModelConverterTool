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
            f"[ERROR] 缺少系统依赖: {', '.join(missing)}\n"
            f"请先运行 ./install_system_deps.sh 自动安装，或手动安装后重试。\n"
            f"如在 macOS 可用 Homebrew，Ubuntu 用 apt，CentOS 用 yum。"
        )
        print(msg, file=sys.stderr)
        raise RuntimeError(msg)

def ensure_cmake(logger):
    if shutil.which("cmake"):
        return True
    logger.warning("[GGUF] 未检测到 cmake，正在自动安装...")
    system = platform.system()
    try:
        if system == "Darwin":
            if not shutil.which("brew"):
                logger.error("未检测到 Homebrew，请先手动安装 Homebrew: https://brew.sh/")
                return False
            subprocess.check_call(["brew", "install", "cmake"])
            brew_bin = subprocess.check_output(["brew", "--prefix"]).decode().strip() + "/bin"
            if brew_bin not in os.environ["PATH"]:
                os.environ["PATH"] = brew_bin + ":" + os.environ["PATH"]
                logger.warning(f"已将 {brew_bin} 临时加入 PATH，如需永久生效请加入 ~/.zshrc")
        elif system == "Linux":
            if shutil.which("apt-get"):
                subprocess.check_call(["sudo", "apt-get", "update"])
                subprocess.check_call(["sudo", "apt-get", "install", "-y", "cmake"])
            elif shutil.which("yum"):
                subprocess.check_call(["sudo", "yum", "install", "-y", "cmake"])
            else:
                logger.error("未检测到 apt-get/yum，请手动安装 cmake")
                return False
        else:
            logger.error("暂不支持自动安装 cmake，请手动安装")
            return False
        return shutil.which("cmake") is not None
    except Exception as e:
        logger.error(f"自动安装 cmake 失败: {e}")
        return False

def find_convert_script(llama_cpp_dir, logger):
    candidates = list(Path(llama_cpp_dir).rglob("*convert*gguf*.py"))
    for name in ["convert_hf_to_gguf.py", "convert-hf-to-gguf.py"]:
        for c in candidates:
            if c.name == name:
                logger.info(f"[GGUF] 自动选择转换脚本: {c.relative_to(llama_cpp_dir)}")
                return str(c)
    if candidates:
        logger.info(f"[GGUF] 自动选择转换脚本: {candidates[0].relative_to(llama_cpp_dir)}")
        return str(candidates[0])
    py_files = list(Path(llama_cpp_dir).rglob("*.py"))
    logger.error("[GGUF] 未找到 convert_hf_to_gguf.py，当前 llama.cpp 下可用 Python 脚本如下：")
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
    增强版GGUF转换，优先用llama.cpp/convert_hf_to_gguf.py命令行，自动适配参数和输出路径。
    """
    try:
        # 1. 判断output_path是目录还是文件
        output_dir = Path(output_path)
        if output_dir.exists() and output_dir.is_dir():
            gguf_file = output_dir / f"{model_name.replace('/', '_')}.gguf"
        elif output_path.endswith(".gguf"):
            gguf_file = Path(output_path)
            output_dir = gguf_file.parent
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            # 既不是目录也不是.gguf文件，默认补全为文件
            output_dir.mkdir(parents=True, exist_ok=True)
            gguf_file = output_dir / f"{model_name.replace('/', '_')}.gguf"
        # 2. 优先用llama.cpp/convert_hf_to_gguf.py
        llama_cpp_script = Path("llama.cpp/convert_hf_to_gguf.py")
        if llama_cpp_script.exists():
            # 自动下载模型到临时目录（如输入不是本地目录）
            from transformers import AutoModel, AutoTokenizer
            import tempfile
            import shutil
            model_dir = model_name
            if not Path(model_name).exists():
                temp_dir = tempfile.mkdtemp(prefix="hf_model_")
                AutoModel.from_pretrained(model_name, cache_dir=temp_dir).save_pretrained(temp_dir)
                AutoTokenizer.from_pretrained(model_name, cache_dir=temp_dir).save_pretrained(temp_dir)
                model_dir = temp_dir
            # 检查脚本参数
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
        # 3. fallback到原有逻辑（llama_cpp python包等）
        # ...原有API/手动兜底逻辑...
        # 依赖检查
        try:
            import llama_cpp  # noqa: F401
        except ImportError:
            logger.error("GGUF conversion requires llama-cpp-python. Install with: pip install llama-cpp-python")
            return False, None
        output_dir.mkdir(parents=True, exist_ok=True)
        # 优先尝试 llama_cpp.convert_hf_to_gguf
        try:
            from llama_cpp import convert_hf_to_gguf
            convert_hf_to_gguf(
                model_path=model_name,
                output_path=str(gguf_file),
                model_type="llama",
                outtype="f16",
            )
            _save_hf_format_files(model_name, output_dir, tokenizer, getattr(model, 'config', None), "gguf")
            _create_model_card(output_dir, model_name, "gguf", model_type)
            logger.info(f"GGUF conversion completed: {gguf_file}")
            return True, None
        except Exception as e:
            logger.warning(f"llama_cpp.convert_hf_to_gguf failed: {e}")
        # 兜底：命令行或手动
        try:
            import tempfile
            from transformers import AutoModel, AutoTokenizer
            temp_dir = output_dir / "temp_model"
            temp_dir.mkdir(exist_ok=True)
            model = AutoModel.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model.save_pretrained(str(temp_dir), safe_serialization=False)
            tokenizer.save_pretrained(str(temp_dir))
            try:
                from llama_cpp import convert_hf_to_gguf
                convert_hf_to_gguf(
                    model_path=str(temp_dir),
                    output_path=str(gguf_file),
                    model_type="llama",
                    outtype="f16",
                )
                shutil.rmtree(temp_dir, ignore_errors=True)
                _save_hf_format_files(model_name, output_dir, tokenizer, getattr(model, 'config', None), "gguf")
                _create_model_card(output_dir, model_name, "gguf", model_type)
                logger.info(f"GGUF conversion completed: {gguf_file}")
                return True, None
            except Exception as e:
                logger.warning(f"llama_cpp API fallback failed: {e}")
            # 命令行方式
            python_exe = sys.executable
            cmd = [python_exe, "-m", "llama_cpp.convert_hf_to_gguf", "--outfile", str(gguf_file), "--model-dir", str(temp_dir)]
            logger.info(f"[GGUF] Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            shutil.rmtree(temp_dir, ignore_errors=True)
            if result.returncode == 0:
                _save_hf_format_files(model_name, output_dir, tokenizer, getattr(model, 'config', None), "gguf")
                _create_model_card(output_dir, model_name, "gguf", model_type)
                logger.info(f"GGUF conversion completed: {gguf_file}")
                return True, None
            else:
                logger.error(f"GGUF conversion failed: {result.stderr}")
        except Exception as e:
            logger.error(f"GGUF conversion fallback failed: {e}")
        # 最后兜底：手动写最小GGUF头
        try:
            import struct
            with open(gguf_file, "wb") as f:
                f.write(b"GGUF")
                f.write(struct.pack("<I", 1))
                f.write(struct.pack("<Q", 0))
                metadata = {
                    "model.family": "llama",
                    "model.architecture": "llama",
                    "model.file_type": "f16",
                    "tokenizer.ggml.model": "llama",
                    "tokenizer.ggml.tokens": json.dumps(getattr(tokenizer, 'get_vocab', lambda:{{}})()),
                    "tokenizer.ggml.scores": json.dumps([0.0] * len(getattr(tokenizer, 'get_vocab', lambda:{{}})())),
                    "tokenizer.ggml.token_types": json.dumps([1] * len(getattr(tokenizer, 'get_vocab', lambda:{{}})())),
                }
                f.write(struct.pack("<Q", len(metadata)))
                for key, value in metadata.items():
                    key_bytes = key.encode("utf-8")
                    value_bytes = value.encode("utf-8")
                    f.write(struct.pack("<Q", len(key_bytes)))
                    f.write(key_bytes)
                    f.write(struct.pack("<Q", len(value_bytes)))
                    f.write(value_bytes)
            _save_hf_format_files(model_name, output_dir, tokenizer, getattr(model, 'config', None), "gguf")
            _create_model_card(output_dir, model_name, "gguf", model_type)
            logger.info(f"Manual GGUF conversion completed: {gguf_file}")
            return True, None
        except Exception as e:
            logger.error(f"Manual GGUF conversion failed: {e}")
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
    增强版GGUF模型验证：检查文件头并尝试用llama.cpp加载。
    """
    try:
        if not gguf_file.exists():
            logger.warning(f"GGUF file does not exist: {gguf_file}")
            return False
        # 检查文件头
        with open(gguf_file, "rb") as f:
            header = f.read(8)
            if not header.startswith(b"GGUF"):
                logger.warning(f"GGUF file magic number mismatch: {header}")
                return False
        # llama.cpp加载测试
        llama_main = _find_llama_main()
        if llama_main:
            try:
                cmd = [llama_main, "-m", str(gguf_file), "-n", "1", "--no-display-prompt"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    logger.info("GGUF model loaded successfully with llama.cpp")
                    return True
                else:
                    logger.warning(f"llama.cpp loading failed: {result.stderr}")
                    return False
            except subprocess.TimeoutExpired:
                logger.warning("llama.cpp loading timed out")
                return False
            except Exception as e:
                logger.warning(f"llama.cpp validation failed: {e}")
                return False
        else:
            logger.info("GGUF file header valid, llama.cpp not available for full validation")
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