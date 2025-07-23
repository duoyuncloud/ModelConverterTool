import logging
from pathlib import Path
from typing import Any
import subprocess
import sys
import os
import tempfile
import shutil

logger = logging.getLogger(__name__)

MLX_EXAMPLES_REPO = "https://github.com/ml-explore/mlx-examples.git"
MLX_EXAMPLES_DIR = "tools/mlx-examples"
REQUIRED_SCRIPTS = ["llms/llama/convert.py"]


def ensure_mlx_examples_available():
    """
    Ensure required MLX conversion scripts are available locally. Auto-clone if missing.
    """
    repo_dir = Path(MLX_EXAMPLES_DIR)
    missing = [s for s in REQUIRED_SCRIPTS if not (repo_dir / s).exists()]
    if not repo_dir.exists() or missing:
        logger.warning(
            f"[MLX] Required MLX conversion scripts not found in {MLX_EXAMPLES_DIR}. Missing: {', '.join(missing) if missing else 'all scripts'}\nAttempting to auto-clone the official mlx-examples repo..."
        )
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        try:
            subprocess.check_call(["git", "clone", "--depth=1", MLX_EXAMPLES_REPO, str(repo_dir)])
        except Exception as e:
            logger.error(f"[MLX] Auto-clone failed: {e}")
            raise RuntimeError("MLX conversion cannot proceed without required scripts.")
        missing = [s for s in REQUIRED_SCRIPTS if not (repo_dir / s).exists()]
        if missing:
            logger.error(
                f"[MLX] Even after auto-clone, missing scripts: {', '.join(missing)}. Please check the official repo or copy the required scripts manually."
            )
            raise RuntimeError("MLX conversion cannot proceed without required scripts.")
        else:
            logger.info(f"[MLX] Successfully cloned mlx-examples. Using scripts in {MLX_EXAMPLES_DIR}.")
    else:
        logger.info(f"[MLX] Using local MLX conversion scripts in {MLX_EXAMPLES_DIR}.")


def convert_to_mlx(
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
    Convert a HuggingFace model to MLX format using the official mlx-lm conversion script.
    The output is always a directory.
    Returns (success: bool, extra_info: dict or None)
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_output = os.path.join(tmpdir, "mlx-out")
            cmd = [sys.executable, "-m", "mlx_lm.convert", "--hf-path", model_name, "--mlx-path", tmp_output]
            if quantization:
                cmd += ["-q", quantization]
            subprocess.check_call(cmd)
            if os.path.isdir(tmp_output):
                if os.path.exists(output_path):
                    shutil.rmtree(output_path)
                shutil.move(tmp_output, output_path)
                return True, None
            raise FileNotFoundError(f"No output directory found at {tmp_output}")
    except Exception as e:
        logger.error(f"MLX conversion failed: {e}")
        return False, {"error": str(e)}


def can_infer_mlx_file(path: str, *args, **kwargs) -> bool:
    """
    Perform dummy inference to test MLX model usability. Returns True if inference is possible, False otherwise.
    """
    try:
        from mlx_lm import load, generate

        if not os.path.isdir(path):
            raise ValueError(f"MLX check expects a directory, got: {path}")
        model, tokenizer = load(path)
        _ = generate(model, tokenizer, prompt="Hello world", verbose=False, max_tokens=1)
        return True
    except Exception as e:
        print(f"[MLX] dynamic check failed: {e}")
        return False


def validate_mlx_file(path: str, *args, **kwargs) -> bool:
    """
    Validate MLX files by checking for config and weights. Returns True if valid, False otherwise.
    """
    if not os.path.isdir(path):
        return False
    files = os.listdir(path)
    has_config = "config.json" in files
    has_weights = any(f.endswith(".npz") or f.endswith(".safetensors") for f in files)
    return has_config and has_weights
