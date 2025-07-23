"""
Configuration management for the model converter tool (simplified for DRY/KISS/YAGNI)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class ConversionConfig:
    input_format: str
    output_format: str
    model_type: str = "auto"
    quantization: Optional[str] = None
    quantization_config: Optional[dict] = None
    device: str = "auto"
    batch_size: int = 1
    max_length: Optional[int] = None
    # Only keep essential fields
    extra: Dict[str, Any] = field(default_factory=dict)


class ConfigManager:
    GLOBAL_CONFIG_PATH = Path.home() / ".model_converter_tool_config.yaml"
    DEFAULT_GLOBAL_CONFIG = {
        "cache_dir": str(Path.home() / ".cache" / "huggingface" / "hub"),
        "default_output_dir": str(Path.home() / "model_converter_tool" / "outputs"),
    }

    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.presets_file = self.config_dir / "model_presets.yaml"
        self._ensure_default_presets()
        self.global_config = self._load_global_config()

    def _ensure_default_presets(self):
        if not self.presets_file.exists():
            self._create_default_presets()

    def _create_default_presets(self):
        default_presets = {
            "common_models": {
                "gpt2": {
                    "description": "GPT-2 base model",
                    "model_type": "text-generation",
                    "default_format": "onnx",
                    "supported_formats": ["onnx", "gguf", "mlx", "torchscript"],
                },
                "bert-base-uncased": {
                    "description": "BERT base uncased model",
                    "model_type": "text-classification",
                    "default_format": "onnx",
                    "supported_formats": ["onnx", "gguf", "mlx", "torchscript"],
                },
            },
        }
        with open(self.presets_file, "w") as f:
            yaml.dump(default_presets, f, default_flow_style=False, indent=2)

    def _load_global_config(self):
        if not self.GLOBAL_CONFIG_PATH.exists():
            self._save_global_config(self.DEFAULT_GLOBAL_CONFIG)
            return self.DEFAULT_GLOBAL_CONFIG.copy()
        try:
            with open(self.GLOBAL_CONFIG_PATH, "r") as f:
                data = yaml.safe_load(f)
                if not isinstance(data, dict):
                    return self.DEFAULT_GLOBAL_CONFIG.copy()
                merged = self.DEFAULT_GLOBAL_CONFIG.copy()
                merged.update(data)
                return merged
        except Exception:
            return self.DEFAULT_GLOBAL_CONFIG.copy()

    def _save_global_config(self, config_dict):
        with open(self.GLOBAL_CONFIG_PATH, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    def all(self):
        return self.global_config.copy()

    def get(self, key):
        return self.global_config.get(key, None)

    def set(self, key, value):
        self.global_config[key] = value
        self._save_global_config(self.global_config)
        return value

    def load_presets(self) -> Dict[str, Any]:
        with open(self.presets_file, "r") as f:
            return yaml.safe_load(f)

    def get_preset(self, preset_name: str) -> Optional[Dict[str, Any]]:
        presets = self.load_presets()
        for category in presets.values():
            if preset_name in category:
                return category[preset_name]
        return None

    def list_presets(self) -> Dict[str, List[str]]:
        presets = self.load_presets()
        return {category: list(models.keys()) for category, models in presets.items()}


def load_config_preset(preset_name: str) -> Optional[Dict[str, Any]]:
    config_manager = ConfigManager()
    return config_manager.get_preset(preset_name)


def load_custom_config(config_file: str) -> Dict[str, Any]:
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = base.copy()
    for k, v in override.items():
        if v is not None:
            merged[k] = v
    return merged


def resolve_final_config(
    preset_name: Optional[str], config_file: Optional[str], cli_overrides: Dict[str, Any]
) -> Dict[str, Any]:
    if config_file:
        base = load_custom_config(config_file)
    elif preset_name:
        base = load_config_preset(preset_name)
    else:
        base = {}
    final = merge_config(base, cli_overrides)
    return final
