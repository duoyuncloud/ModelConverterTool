"""
Utility functions for model conversion (simplified for DRY/KISS/YAGNI)
"""

import json
import logging
import os
from pathlib import Path
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()


def ensure_output_directory(output_path: str) -> str:
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_path)


def get_local_cache_path(model_name: str) -> str:
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_cache_dir = cache_dir / f"models--{model_name.replace('/', '--')}"
    if not model_cache_dir.exists():
        return model_name
    snapshots_dir = model_cache_dir / "snapshots"
    if not snapshots_dir.exists():
        return model_name
    snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
    if not snapshot_dirs:
        return model_name
    latest_snapshot = max(snapshot_dirs, key=lambda x: x.stat().st_mtime)
    required_files = ["config.json", "model.safetensors", "tokenizer.json"]
    if all((latest_snapshot / f).exists() for f in required_files):
        logger.info(f"Found local cache: {latest_snapshot}")
        return str(latest_snapshot)
    return model_name


def load_model_and_tokenizer(
    model_name: str,
    model_type: str = "auto",
    fake_weight: bool = False,
    fake_weight_shape_dict: dict = None,
    model=None,
    tokenizer=None,
    **kwargs,
):
    """
    Unified model and tokenizer loading, with cache and fake weight support.
    """
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

    kwargs.setdefault("trust_remote_code", True)
    if fake_weight:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        if hasattr(config, "quantization_config"):
            delattr(config, "quantization_config")
        if hasattr(config, "to_dict") and "quantization_config" in config.to_dict():
            config_dict = config.to_dict()
            config_dict.pop("quantization_config", None)
            config = type(config).from_dict(config_dict)
        model_class = (
            AutoModelForCausalLM if getattr(config, "model_type", None) and ("qwen" in config.model_type) else AutoModel
        )
        model = generate_fake_model(config, model_class, fake_weight_shape_dict)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        return model, tokenizer
    # Normal loading
    if model is None:
        try:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            model_type_val = getattr(config, "model_type", None)
            model_class = (
                AutoModelForCausalLM
                if model_type_val
                and (
                    "causal" in model_type
                    or "lm" in model_type
                    or "generation" in model_type
                    or "text-generation" in model_type
                )
                else AutoModel
            )
        except Exception:
            model_class = AutoModel
        local_path = get_local_cache_path(model_name)
        try:
            logger.info(f"Attempting to load model from local cache: {local_path}")
            model = model_class.from_pretrained(local_path, local_files_only=True, **kwargs)
        except Exception:
            logger.warning(f"Local cache incomplete, attempting to load from network: {model_name}")
            model = model_class.from_pretrained(model_name, **kwargs)
    if tokenizer is None:
        local_path = get_local_cache_path(model_name)
        try:
            logger.info(f"Attempting to load tokenizer from local cache: {local_path}")
            tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True, **kwargs)
        except Exception:
            logger.warning(f"Local cache incomplete, attempting to load tokenizer from network: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    return model, tokenizer


def get_calibration_dataset(use_large_calibration, tag="AWQ"):
    if use_large_calibration:
        try:
            from datasets import load_dataset

            ds = load_dataset("openwebtext", split="train", trust_remote_code=True)
            calibration_dataset = [x["text"] for x in ds.select(range(1000)) if len(x["text"].split()) > 32]
            if len(calibration_dataset) < 1000:
                calibration_dataset += ["The quick brown fox jumps over the lazy dog."] * (
                    1000 - len(calibration_dataset)
                )
            logger.info(
                f"[{tag}] Using HuggingFace openwebtext sampling {len(calibration_dataset)} high-quality calibration texts"
            )
            return calibration_dataset
        except Exception as e:
            logger.warning(
                f"[{tag}] Failed to load high-quality calibration set, falling back to built-in samples: {e}"
            )
            return [
                "The quick brown fox jumps over the lazy dog. " * 20,
                f"{tag} high-precision calibration sentence. " * 20,
                "This is a long calibration text for high-precision quantization. " * 20,
            ]
    else:
        return [
            "This is a much longer calibration sentence that should have more than ten tokens for the quantization process.",
            "Another example of a calibration sentence that is sufficiently long to pass the minimum length requirement for quantization.",
            "Quantization calibration requires sentences that are not too short, so this one is also long enough to be valid for the test.",
        ]


def generate_fake_model(config, model_class, fake_weight_shape_dict: dict = None):
    import torch

    model = model_class.from_config(config)
    for name, param in model.named_parameters():
        if param.requires_grad:
            if fake_weight_shape_dict and name in fake_weight_shape_dict:
                try:
                    shape = tuple(fake_weight_shape_dict[name])
                    if shape != tuple(param.shape):
                        print(
                            f"[WARNING] Shape mismatch for {name}: config {shape}, model default {tuple(param.shape)}. Using config shape."
                        )
                    param.data = torch.zeros(shape, dtype=param.dtype, device=param.device)
                except Exception as e:
                    print(
                        f"[WARNING] Could not set fake weight for {name} with shape {fake_weight_shape_dict[name]}: {e}"
                    )
            else:
                torch.nn.init.zeros_(param)
    return model


def create_dummy_model(output_dir: str, **kwargs):
    os.makedirs(output_dir, exist_ok=True)
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 128,
        "intermediate_size": 512,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "vocab_size": 1000,
        "max_position_embeddings": 512,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "torch_dtype": "float16",
        **kwargs,
    }
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "model_max_length": 512,
    }
    with open(os.path.join(output_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, indent=2)
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
    for i in range(4, 100):
        vocab[f"token{i}"] = i
    with open(os.path.join(output_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)
    logger.info(f"✅ Simple dummy model config generated at {output_dir}")
    logger.warning("⚠️  This is a test-only dummy model. Use real models in production.")


def auto_complete_output_path(input_path, output_path, to_format):
    to_format = {"hf": "huggingface"}.get(to_format.lower(), to_format.lower())
    file_exts = {
        "onnx": ".onnx",
        "gguf": ".gguf",
        "pt": ".pt",
        "torchscript": ".pt",
        "safetensors": ".safetensors",
        "fp16": ".safetensors",
    }
    base = os.path.basename(input_path)

    def to_dir_name(path, ext=None):
        p = Path(path)
        if ext and p.name.endswith(ext):
            return str(p.with_suffix("")) + f"_{to_format}"
        if p.suffix:
            return str(p.with_suffix("")) + f"_{to_format}"
        return str(p)

    if not output_path:
        return f"./outputs/{base}_{to_format}"
    if os.path.isdir(output_path):
        return output_path
    for ext in file_exts.values():
        if output_path.endswith(ext):
            return to_dir_name(output_path, ext)
    if not os.path.exists(output_path) and not output_path.endswith("/"):
        return to_dir_name(output_path)
    return output_path


def auto_load_model_and_tokenizer(model, tokenizer, model_name, model_type):
    """
    Backward-compatible loader for engine modules. Prefer load_model_and_tokenizer, but keep this for engine compatibility.
    """

    def _load(model_type_to_use):
        loaded_model, loaded_tokenizer = load_model_and_tokenizer(
            model_name,
            model_type=model_type_to_use,
            model=model,
            tokenizer=tokenizer,
        )
        return loaded_model, loaded_tokenizer

    try:
        return _load(model_type)
    except Exception as e:
        if model_type != "auto":
            try:
                return _load("auto")
            except Exception:
                raise e
        else:
            raise


def patch_quantization_config_file(config_path: Path, bits: int, group_size: int, sym: bool, desc: str = None):
    """
    Patch quantization-related fields in a config.json file. Required for quantization engine compatibility.
    """
    if not config_path.exists():
        logger.warning(f"Config file not found for patching: {config_path}")
        return
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        config["quantization_bits"] = bits
        config["quantization_group_size"] = group_size
        config["quantization_symmetric"] = sym
        if desc is not None:
            config["quantization_desc"] = desc
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Patched quantization config in {config_path}")
    except Exception as e:
        logger.error(f"Failed to patch quantization config: {e}")
