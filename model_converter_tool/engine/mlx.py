import logging
from pathlib import Path
from typing import Any
import subprocess
import sys
import os
import json
from datetime import datetime
from model_converter_tool.utils import auto_load_model_and_tokenizer, patch_quantization_config

logger = logging.getLogger(__name__)

MLX_EXAMPLES_REPO = "https://github.com/ml-explore/mlx-examples.git"
MLX_EXAMPLES_DIR = "examples/mlx"
CONVERT_SCRIPT = "convert_checkpoint.py"

def _auto_setup_mlx_examples():
    repo_dir = Path(MLX_EXAMPLES_DIR)
    def clone_repo():
        if repo_dir.exists():
            import shutil
            shutil.rmtree(repo_dir)
        logger.info("[MLX] Cloning latest mlx-examples...")
        subprocess.check_call(["git", "clone", MLX_EXAMPLES_REPO, str(repo_dir)])
    try:
        if not repo_dir.exists():
            clone_repo()
        convert_scripts = list(repo_dir.rglob("*convert*.py"))
        if not convert_scripts:
            py_files = list(repo_dir.rglob("*.py"))
            logger.error("[MLX] No scripts containing 'convert' found. Available Python scripts in current repository:")
            for f in py_files:
                logger.error(f"    - {f.relative_to(repo_dir)}")
            raise RuntimeError(
                f"[MLX] No conversion scripts found, possibly due to official repository structure changes or network/proxy issues. Please check the script list above, or manually adapt the latest conversion script. Repository address: {MLX_EXAMPLES_REPO}"
            )
        root_scripts = [s for s in convert_scripts if s.parent == repo_dir]
        script_path = root_scripts[0] if root_scripts else convert_scripts[0]
        logger.info(f"[MLX] Automatically selected conversion script: {script_path.relative_to(repo_dir)}")
        if script_path.name == "convert.py":
            script_type = "convert_py"
        elif script_path.name == "convert_checkpoint.py":
            script_type = "convert_checkpoint"
        else:
            script_type = "unknown"
        return str(script_path), script_type
    except Exception as e:
        logger.error(f"[MLX] setup failed: {e}")
        raise

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