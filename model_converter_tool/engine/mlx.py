import logging
from pathlib import Path
from typing import Any
import subprocess
import sys
import os
import json
from datetime import datetime
from model_converter_tool.utils import auto_load_model_and_tokenizer, patch_quantization_config
import shutil

logger = logging.getLogger(__name__)

MLX_EXAMPLES_REPO = "https://github.com/ml-explore/mlx-examples.git"
MLX_EXAMPLES_DIR = "tools/mlx-examples"
REQUIRED_SCRIPTS = ["llms/llama/convert.py"]  # Only keep what is actually needed


def ensure_mlx_examples_available():
    """
    Ensure that the required MLX conversion scripts are available locally.
    If missing, auto-clone the official mlx-examples repo into examples/mlx.
    Only error if clone fails or scripts are still missing.
    """
    repo_dir = Path(MLX_EXAMPLES_DIR)
    missing = [s for s in REQUIRED_SCRIPTS if not (repo_dir / s).exists()]
    if not repo_dir.exists() or missing:
        logger.warning(
            f"[MLX] Required MLX conversion scripts not found in {MLX_EXAMPLES_DIR}.\n"
            f"Missing: {', '.join(missing) if missing else 'all scripts'}\n"
            f"Attempting to auto-clone the official mlx-examples repo..."
        )
        # Remove old dir if exists (to avoid partial/corrupt state)
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        try:
            subprocess.check_call(["git", "clone", "--depth=1", MLX_EXAMPLES_REPO, str(repo_dir)])
        except Exception as e:
            logger.error(f"[MLX] Auto-clone failed: {e}")
            raise RuntimeError("MLX conversion cannot proceed without required scripts.")
        # Re-check after clone
        missing = [s for s in REQUIRED_SCRIPTS if not (repo_dir / s).exists()]
        if missing:
            logger.error(
                f"[MLX] Even after auto-clone, missing scripts: {', '.join(missing)}.\n"
                f"Please check the official repo or copy the required scripts manually."
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
    quantization_config: dict = None
) -> tuple:
    """
    Convert a HuggingFace model to MLX format using the official mlx-lm conversion script.
    Uses a unique temporary output path to avoid path conflicts, then moves the result to the user-specified output directory.
    The output is always a directory (never a .npz file).
    """
    import subprocess
    import sys
    import os
    import tempfile
    import shutil
    try:
        # Use a unique temporary directory for mlx-lm output
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_output = os.path.join(tmpdir, "mlx-out")
            cmd = [
                sys.executable, '-m', 'mlx_lm.convert',
                '--hf-path', model_name,
                '--mlx-path', tmp_output
            ]
            if quantization:
                cmd += ['-q', quantization]
            subprocess.check_call(cmd)
            # Print the contents of the temp directory for debugging
            print("[DEBUG] Contents of tempdir after mlx-lm.convert:", os.listdir(tmpdir))
            # Move/rename the output directory to user-specified output_path
            if os.path.isdir(tmp_output):
                # Remove existing output_path if it exists
                if os.path.exists(output_path):
                    shutil.rmtree(output_path)
                shutil.move(tmp_output, output_path)
                return True, None
            # If not, fail
            raise FileNotFoundError(f"No output directory found at {tmp_output}")
    except Exception as e:
        logger.error(f"MLX conversion failed: {e}")
        return False, {"error": str(e)}


def can_infer_mlx_file(path: str, *args, **kwargs) -> bool:
    """
    Dynamic check for MLX files. Loads the model and tokenizer with mlx_lm and runs a real dummy inference.
    Expects path to be a directory containing MLX weights/config.
    Returns True if inference is possible, False otherwise.
    """
    try:
        import os
        from mlx_lm import load, generate
        # path should be a directory containing MLX weights/config
        if not os.path.isdir(path):
            raise ValueError(f"MLX check expects a directory, got: {path}")
        model, tokenizer = load(path)
        _ = generate(model, tokenizer, prompt="Hello world", verbose=False, max_tokens=1)
        return True
    except Exception as e:
        logger.error(f"MLX dynamic check failed: {e}")
        return False

def validate_mlx_file(path: str, *args, **kwargs) -> bool:
    import os
    if not os.path.isdir(path):
        return False
    files = os.listdir(path)
    has_config = 'config.json' in files
    has_weights = any(f.endswith('.npz') or f.endswith('.safetensors') for f in files)
    return has_config and has_weights 