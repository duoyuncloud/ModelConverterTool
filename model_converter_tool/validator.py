"""
Model validation functionality
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of model validation"""

    is_valid: bool
    details: List[str]
    errors: List[str]
    warnings: List[str]
    model_info: Dict[str, Any]


class ModelValidator:
    """Validates model files and configurations"""

    def __init__(self):
        self.required_files = {
            "huggingface": ["config.json", "pytorch_model.bin", "tokenizer.json"],
            "onnx": ["model.onnx"],
            "gguf": ["model.gguf"],
            "mlx": ["model.npz", "config.json"],
            "torchscript": ["model.pt"],
        }

    def validate_model(
        self,
        model_path: str,
        model_type: str = "auto",
        check_weights: bool = False,
        check_config: bool = False,
    ) -> ValidationResult:
        """
        Validate a model

        Args:
            model_path: Path to model (hf:model_name or local path)
            model_type: Model type for validation
            check_weights: Whether to check model weights
            check_config: Whether to check model configuration

        Returns:
            ValidationResult: Validation result with details
        """
        details = []
        errors = []
        warnings = []
        model_info = {}
        detected_format = None
        try:
            # Parse model path
            if model_path.startswith("hf:"):
                model_name = model_path[3:]
                input_type = "huggingface"
                details.append(f"Validating HuggingFace model: {model_name}")
            else:
                model_name = model_path
                input_type = "local"
                details.append(f"Validating local model: {model_name}")
            # Basic validation
            if input_type == "local":
                if not os.path.exists(model_name):
                    errors.append(f"Model path does not exist: {model_name}")
                    return ValidationResult(
                        False, details, errors, warnings, model_info
                    )
                # Smart format detection
                detected_format, detect_details = self._detect_format(model_name)
                details.extend(detect_details)
                # If user did not specify model_type, use detected_format
                effective_type = model_type
                if model_type == "auto" and detected_format:
                    effective_type = detected_format
                    details.append(f"🔎 Detected model format: {detected_format}")
                elif model_type == "auto":
                    details.append(
                        "⚠️ Could not confidently detect model format, fallback to HuggingFace."
                    )
                    effective_type = "huggingface"
                # Check required files
                file_validation = self._validate_files(model_name, effective_type)
                details.extend(file_validation["details"])
                errors.extend(file_validation["errors"])
                warnings.extend(file_validation["warnings"])
            # Configuration validation
            if check_config:
                config_validation = self._validate_config(model_path, model_type)
                details.extend(config_validation["details"])
                errors.extend(config_validation["errors"])
                warnings.extend(config_validation["warnings"])
            # Weights validation
            if check_weights:
                weights_validation = self._validate_weights(model_path, model_type)
                details.extend(weights_validation["details"])
                errors.extend(weights_validation["errors"])
                warnings.extend(weights_validation["warnings"])
            # Model info
            model_info = self._extract_model_info(model_path, model_type)
            is_valid = len(errors) == 0
            if is_valid:
                details.append("✅ Model validation passed")
            else:
                details.append("❌ Model validation failed")
            return ValidationResult(is_valid, details, errors, warnings, model_info)
        except Exception as e:
            errors.append(f"Validation error: {e}")
            return ValidationResult(False, details, errors, warnings, model_info)

    def _validate_files(self, model_path: str, model_type: str) -> Dict[str, List[str]]:
        """Validate required files exist"""
        details = []
        errors = []
        warnings = []
        model_dir = Path(model_path)
        # 优化：对 onnx/torchscript/gguf/mlx 只要求主文件存在，附属文件缺失降级为 warning
        # main_file_map = {  # Not used
        #     "onnx": "model.onnx",
        #     "torchscript": "model.pt",
        #     "gguf": "model.gguf",
        #     "mlx": "model.npz",
        # }
        # fallback 逻辑
        key = model_type if model_type in self.required_files else "huggingface"
        # 检查是否真的为 huggingface 格式（有 config.json 且有权重文件）
        is_hf = (model_dir / "config.json").exists() and (
            (model_dir / "pytorch_model.bin").exists()
            or (model_dir / "model.safetensors").exists()
        )
        for file_name in self.required_files[key]:
            file_path = model_dir / file_name
            if file_path.exists():
                details.append(f"✅ Found required file: {file_name}")
            else:
                # For HuggingFace models, check for safetensors as alternative to
                # pytorch_model.bin
                if (
                    file_name == "pytorch_model.bin"
                    and (model_dir / "model.safetensors").exists()
                ):
                    details.append(
                        f"✅ Found required file: model.safetensors (alternative to {file_name})"
                    )
                elif file_name == "pytorch_model.bin" and not is_hf:
                    # 非 huggingface 格式，缺失 pytorch_model.bin 只给 warning
                    warnings.append(
                        f"⚠️ Missing optional file: {file_name} (not required for this format)"
                    )
                else:
                    errors.append(f"❌ Missing required file: {file_name}")

        return {"details": details, "errors": errors, "warnings": warnings}

    def _validate_config(
        self, model_path: str, model_type: str
    ) -> Dict[str, List[str]]:
        """Validate model configuration"""
        details = []
        errors = []
        warnings = []

        try:
            if model_path.startswith("hf:"):
                # For HuggingFace models, can't easily validate config without
                # downloading
                details.append(
                    "ℹ️ HuggingFace model config validation skipped (requires download)"
                )
                return {"details": details, "errors": errors, "warnings": warnings}

            config_path = Path(model_path) / "config.json"
            if not config_path.exists():
                errors.append("❌ Config file not found")
                return {"details": details, "errors": errors, "warnings": warnings}

            # Load and validate config
            with open(config_path, "r") as f:
                config = json.load(f)

            # Basic config validation
            required_keys = ["model_type", "architectures"]
            for key in required_keys:
                if key in config:
                    details.append(f"✅ Config contains {key}: {config[key]}")
                else:
                    warnings.append(f"⚠️ Config missing {key}")

            details.append("✅ Configuration validation completed")

        except json.JSONDecodeError as e:
            errors.append(f"❌ Invalid JSON in config file: {e}")
        except Exception as e:
            errors.append(f"❌ Config validation error: {e}")

        return {"details": details, "errors": errors, "warnings": warnings}

    def _validate_weights(
        self, model_path: str, model_type: str
    ) -> Dict[str, List[str]]:
        """Validate model weights"""
        details = []
        errors = []
        warnings = []

        try:
            if model_path.startswith("hf:"):
                details.append(
                    "ℹ️ HuggingFace model weights validation skipped (requires download)"
                )
                return {"details": details, "errors": errors, "warnings": warnings}

            model_dir = Path(model_path)

            # Check for weight files
            weight_files = list(model_dir.glob("*.bin")) + list(
                model_dir.glob("*.safetensors")
            )

            if weight_files:
                total_size = sum(f.stat().st_size for f in weight_files)
                details.append(f"✅ Found {len(weight_files)} weight files")
                details.append(f"✅ Total weight size: {total_size / (1024**3):.2f} GB")
            else:
                warnings.append("⚠️ No weight files found")

            details.append("✅ Weights validation completed")

        except Exception as e:
            errors.append(f"❌ Weights validation error: {e}")

        return {"details": details, "errors": errors, "warnings": warnings}

    def _extract_model_info(self, model_path: str, model_type: str) -> Dict[str, Any]:
        """Extract basic model information"""
        info = {
            "path": model_path,
            "type": model_type,
            "size": None,
            "architecture": None,
            "parameters": None,
        }

        try:
            if not model_path.startswith("hf:"):
                model_dir = Path(model_path)
                if model_dir.exists():
                    # Calculate directory size
                    total_size = sum(
                        f.stat().st_size for f in model_dir.rglob("*") if f.is_file()
                    )
                    info["size"] = total_size

                    # Try to get architecture info from config
                    config_path = model_dir / "config.json"
                    if config_path.exists():
                        with open(config_path, "r") as f:
                            config = json.load(f)
                        info["architecture"] = config.get("architectures", [None])[0]
                        info["parameters"] = config.get("num_parameters")
        except Exception as e:
            logger.warning(f"Could not extract model info: {e}")

        return info

    def _detect_format(self, model_path: str):
        """Detect model format based on files in the directory. Returns (format, details)"""
        details = []
        model_dir = Path(model_path)
        # Priority: onnx > gguf > mlx > torchscript > huggingface
        if (model_dir / "model.onnx").exists():
            details.append("Detected model.onnx file, likely ONNX format.")
            return "onnx", details
        if (model_dir / "model.gguf").exists():
            details.append("Detected model.gguf file, likely GGUF format.")
            return "gguf", details
        if (model_dir / "model.npz").exists() and (model_dir / "config.json").exists():
            details.append("Detected model.npz and config.json, likely MLX format.")
            return "mlx", details
        if (model_dir / "model.pt").exists():
            details.append("Detected model.pt file, likely TorchScript format.")
            return "torchscript", details
        if (model_dir / "config.json").exists() and (
            (model_dir / "pytorch_model.bin").exists()
            or (model_dir / "model.safetensors").exists()
        ):
            details.append(
                "Detected config.json and weights, likely HuggingFace format."
            )
            return "huggingface", details
        # Fallback
        details.append("No known model format signature found.")
        return None, details
