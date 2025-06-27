"""
Configuration management for model conversion
"""

import os
import yaml
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ConversionConfig:
    """Configuration for model conversion"""

    input_format: str
    output_format: str
    model_type: str = "auto"
    quantization: Optional[str] = None
    device: str = "auto"
    batch_size: int = 1
    max_length: Optional[int] = None
    onnx_opset_version: int = 17
    gptq_bits: int = 4
    awq_bits: int = 4
    gguf_quantization: str = "q4_k_m"
    mlx_quantization: str = "q4_k_m"

    # Model architecture params (like ModelSpeedTest)
    hidden_size: Optional[int] = None
    num_hidden_layers: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    vocab_size: Optional[int] = None
    intermediate_size: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    dim_model_base: Optional[int] = None
    scale_emb: Optional[float] = None
    scale_depth: Optional[float] = None
    sliding_window: Optional[List] = None

    # Advanced conversion options
    use_safe_serialization: bool = True
    trust_remote_code: bool = False
    torch_dtype: str = "auto"
    attention_bias: bool = False
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    initializer_range: float = 0.1
    rms_norm_eps: float = 1e-05
    rope_theta: float = 10000.0
    use_cache: bool = True
    tie_word_embeddings: bool = True

    # Add more as needed
    extra: Dict[str, Any] = field(default_factory=dict)


class ConfigManager:
    """Manages configuration presets and settings"""

    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.presets_file = self.config_dir / "model_presets.yaml"
        self._ensure_default_presets()

    def _ensure_default_presets(self):
        """Ensure default presets exist"""
        if not self.presets_file.exists():
            self._create_default_presets()

    def _create_default_presets(self):
        """Create default model presets"""
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
                "distilbert-base-uncased": {
                    "description": "DistilBERT base uncased model",
                    "model_type": "text-classification",
                    "default_format": "onnx",
                    "supported_formats": ["onnx", "gguf", "mlx", "torchscript"],
                },
                "t5-small": {
                    "description": "T5 small model",
                    "model_type": "text2text-generation",
                    "default_format": "onnx",
                    "supported_formats": ["onnx", "gguf", "mlx", "torchscript"],
                },
            },
            "llm_models": {
                "meta-llama/Llama-2-7b-hf": {
                    "description": "Llama 2 7B model",
                    "model_type": "text-generation",
                    "default_format": "gguf",
                    "supported_formats": ["gguf", "onnx", "mlx"],
                    "quantization_options": ["q4_k_m", "q8_0", "q5_k_m"],
                },
                "microsoft/DialoGPT-medium": {
                    "description": "DialoGPT medium model",
                    "model_type": "text-generation",
                    "default_format": "onnx",
                    "supported_formats": ["onnx", "gguf", "mlx"],
                },
            },
            "vision_models": {
                "microsoft/resnet-50": {
                    "description": "ResNet-50 model",
                    "model_type": "image-classification",
                    "default_format": "onnx",
                    "supported_formats": ["onnx", "mlx", "torchscript"],
                },
                "google/vit-base-patch16-224": {
                    "description": "ViT base model",
                    "model_type": "image-classification",
                    "default_format": "onnx",
                    "supported_formats": ["onnx", "mlx", "torchscript"],
                },
            },
        }

        with open(self.presets_file, "w") as f:
            yaml.dump(default_presets, f, default_flow_style=False, indent=2)

    def load_presets(self) -> Dict[str, Any]:
        """Load model presets from file"""
        with open(self.presets_file, "r") as f:
            return yaml.safe_load(f)

    def get_preset(self, preset_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific preset by name"""
        presets = self.load_presets()

        # Search in all categories
        for category in presets.values():
            if preset_name in category:
                return category[preset_name]

        return None

    def list_presets(self) -> Dict[str, List[str]]:
        """List all available presets by category"""
        presets = self.load_presets()
        return {category: list(models.keys()) for category, models in presets.items()}


def load_config_preset(preset_name: str) -> Optional[Dict[str, Any]]:
    """Load configuration from a preset"""
    config_manager = ConfigManager()
    preset = config_manager.get_preset(preset_name)

    if preset is None:
        return None

    return preset


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
    preset_name: Optional[str],
    config_file: Optional[str],
    cli_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    # 1. Load preset or custom config
    if config_file:
        base = load_custom_config(config_file)
    elif preset_name:
        base = load_config_preset(preset_name)
    else:
        base = {}
    # 2. Merge CLI overrides
    final = merge_config(base, cli_overrides)
    return final


def list_available_presets():
    """List all available model presets"""
    config_manager = ConfigManager()
    presets = config_manager.list_presets()

    print("📋 Available Model Presets:")
    print("=" * 50)

    for category, model_list in presets.items():
        print(f"\n🔹 {category.replace('_', ' ').title()}:")
        for model_name in model_list:
            preset = config_manager.get_preset(model_name)
            if preset:
                print(f"  - {model_name}")
                print(f"    Description: {preset.get('description', 'N/A')}")
                print(f"    Type: {preset.get('model_type', 'auto')}")
                print(f"    Default Format: {preset.get('default_format', 'onnx')}")
                print(
                    f"    Supported Formats: {', '.join(preset.get('supported_formats', []))}"
                )
                if preset.get("quantization_options"):
                    print(
                        f"    Quantization: {', '.join(preset['quantization_options'])}"
                    )
                print()


def create_batch_config_template():
    """Create a template for batch conversion configuration"""
    template = {
        "models": [
            {
                "name": "gpt2_to_onnx",
                "input": "gpt2",
                "output_format": "onnx",
                "output_path": "outputs/gpt2.onnx",
                "model_type": "text-generation",
            },
            {
                "name": "bert_to_gguf",
                "input": "bert-base-uncased",
                "output_format": "gguf",
                "output_path": "outputs/bert.gguf",
                "model_type": "text-classification",
                "quantization": "q4_k_m",
            },
        ]
    }

    config_path = Path("configs/batch_template.yaml")
    config_path.parent.mkdir(exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(template, f, default_flow_style=False, indent=2)

    print(f"✅ Batch configuration template created: {config_path}")
    return config_path
