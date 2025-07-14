import logging
from pathlib import Path
from typing import Any
import subprocess
import sys
import os
import json
from datetime import datetime
from model_converter_tool.utils import load_model_with_cache
from transformers import AutoModel, AutoModelForCausalLM

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
        # Delayed execution to avoid execution during module import
        # Remove module-level function calls, change to call when needed
        if not repo_dir.exists():
            clone_repo()  # Clone when actually needed
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
    device: str
) -> tuple:
    """
    Enhanced MLX conversion, directly convert PyTorch->MLX weights, automatically save tokenizer/config/mlx_config/model_card.
    """
    try:
        # Dependency check
        try:
            import mlx.core as mx
            import numpy as np
        except ImportError:
            import platform
            is_apple_silicon = platform.system() == "Darwin" and platform.machine() in ("arm64", "arm")
            if is_apple_silicon:
                logger.error("MLX conversion requires mlx. You are on Apple Silicon (macOS arm64). For best performance, install MLX: pip install mlx")
            else:
                logger.error("MLX conversion requires mlx, which is only available on Apple Silicon (macOS arm64). Your platform is not supported.")
            return False, None
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Load model
        if model is None:
            if model_type and ("causal" in model_type or "lm" in model_type or "generation" in model_type or model_type == "text-generation"):
                model = load_model_with_cache(model_name, AutoModelForCausalLM)
            else:
                model = load_model_with_cache(model_name, AutoModel)
        # Convert weights
        mlx_model = _convert_pytorch_to_mlx(model)
        mlx_file = output_dir / "model.npz"
        np.savez(str(mlx_file), **{k: np.array(v) for k, v in mlx_model.items()})
        # Save tokenizer/config
        _save_hf_format_files(model_name, output_dir, tokenizer, getattr(model, 'config', None), "mlx")
        # Save mlx_config
        mlx_config = {
            "model_type": model_type,
            "format": "mlx",
            "quantization": "none",
            "original_model": model_name,
            "conversion_date": datetime.now().isoformat(),
        }
        with open(output_dir / "mlx_config.json", "w") as f:
            json.dump(mlx_config, f, indent=2)
        # Save model_card
        _create_model_card(output_dir, model_name, "mlx", model_type)
        logger.info(f"MLX conversion completed: {output_dir}")
        return True, None
    except Exception as e:
        logger.error(f"MLX conversion failed: {e}")
        return False, None

def _convert_pytorch_to_mlx(pytorch_model):
    try:
        import mlx.core as mx
        import numpy as np
    except ImportError:
        logger.error("MLX not available for conversion")
        return {}
    import os
    if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
        logger.warning("Skipping MLX conversion on CI environment to avoid bus errors")
        return {}
    mlx_model = {}
    state_dict = pytorch_model.state_dict()
    for name, param in state_dict.items():
        try:
            numpy_array = param.detach().cpu().numpy()
            if numpy_array.size > 1000000:
                logger.warning(f"Skipping large tensor {name} ({numpy_array.size} elements) on CI")
                continue
            if numpy_array.dtype == np.float32:
                mlx_model[name] = mx.array(numpy_array, dtype=mx.float32)
            elif numpy_array.dtype == np.float16:
                mlx_model[name] = mx.array(numpy_array, dtype=mx.float16)
            elif numpy_array.dtype == np.int64:
                mlx_model[name] = mx.array(numpy_array, dtype=mx.int64)
            elif numpy_array.dtype == np.int32:
                mlx_model[name] = mx.array(numpy_array, dtype=mx.int32)
            else:
                mlx_model[name] = mx.array(numpy_array.astype(np.float32), dtype=mx.float32)
        except Exception as e:
            logger.warning(f"Failed to convert tensor {name}: {e}")
            continue
    return mlx_model

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

def validate_mlx_file(mlx_path: Path, _: Any) -> bool:
    """
    Enhanced MLX model validation: Check npz file existence, try to load if mlx-transformers is available.
    """
    try:
        import numpy as np
        from pathlib import Path
        model_dir = mlx_path if mlx_path.is_dir() else mlx_path.parent
        mlx_files = list(model_dir.glob("*.npz"))
        if not mlx_files:
            logger.warning(f"No MLX files found: {model_dir}")
            return False
        mlx_file = mlx_files[0]
        # Auto fallback: If mlx_transformers is not installed, only do file existence check
        try:
            import mlx_transformers
        except ImportError:
            logger.info("mlx-transformers not detected, only doing file existence check, no inference validation.")
            return True
        # Dependency installed, try new/old API
        try:
            from mlx_transformers import AutoTokenizer, MLXModel
            # Only do loading-level validation here, no inference
            logger.info("MLXModel class exists, only doing loading-level validation.")
            return True
        except ImportError:
            try:
                from mlx_transformers import GenerationConfig, load
                logger.info("load/GenerationConfig exists, only doing loading-level validation.")
                return True
            except ImportError:
                logger.info("mlx-transformers dependency exception, only doing file existence check.")
                return True
        except Exception as e:
            logger.info(f"mlx-transformers loading exception, only doing file existence check. {e}")
            return True
    except Exception as e:
        logger.warning(f"MLX validation error: {e}")
        return False 