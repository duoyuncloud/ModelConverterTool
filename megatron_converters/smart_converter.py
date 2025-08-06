#!/usr/bin/env python3
"""
Smart Conversion Scheduler
Automatically selects the best conversion script based on model type, size, and parallel strategy
"""

import torch
from typing import Dict, Optional, Tuple
from pathlib import Path


class SmartConverter:
    """Smart conversion scheduler for automatic model conversion"""

    def __init__(self):
        self.model_configs = {
            # MiniCPM series
            "minicpm": {
                "0.5b": {
                    "layers": 12,
                    "tp_size": 1,
                    "pp_size": 1,
                    "use_basic": True,
                    "num_kv_heads": 8,
                    "num_query_heads": 32,
                },
                "1.5b": {
                    "layers": 18,
                    "tp_size": 1,
                    "pp_size": 1,
                    "use_basic": True,
                    "num_kv_heads": 8,
                    "num_query_heads": 32,
                },
                "3b": {
                    "layers": 24,
                    "tp_size": 1,
                    "pp_size": 1,
                    "use_basic": True,
                    "num_kv_heads": 8,
                    "num_query_heads": 32,
                },
                "8b": {
                    "layers": 32,
                    "tp_size": 2,
                    "pp_size": 1,
                    "use_basic": False,
                    "num_kv_heads": 8,
                    "num_query_heads": 32,
                },
                "14b": {
                    "layers": 40,
                    "tp_size": 4,
                    "pp_size": 1,
                    "use_basic": False,
                    "num_kv_heads": 8,
                    "num_query_heads": 32,
                },
            },
            # Llama series
            "llama": {
                "0.5b": {
                    "layers": 12,
                    "tp_size": 1,
                    "pp_size": 1,
                    "use_basic": True,
                    "num_kv_heads": 8,
                    "num_query_heads": 32,
                },
                "7b": {
                    "layers": 32,
                    "tp_size": 1,
                    "pp_size": 1,
                    "use_basic": True,
                    "num_kv_heads": 8,
                    "num_query_heads": 32,
                },
                "13b": {
                    "layers": 40,
                    "tp_size": 2,
                    "pp_size": 1,
                    "use_basic": False,
                    "num_kv_heads": 8,
                    "num_query_heads": 40,
                },
                "30b": {
                    "layers": 60,
                    "tp_size": 4,
                    "pp_size": 1,
                    "use_basic": False,
                    "num_kv_heads": 8,
                    "num_query_heads": 52,
                },
                "65b": {
                    "layers": 80,
                    "tp_size": 8,
                    "pp_size": 1,
                    "use_basic": False,
                    "num_kv_heads": 8,
                    "num_query_heads": 64,
                },
            },
        }

    def detect_model_type_and_size(self, checkpoint_path: str) -> Tuple[str, str]:
        """
        Automatically detect model type and size from checkpoint structure

        Args:
            checkpoint_path: Path to the checkpoint directory or HuggingFace model name

        Returns:
            Tuple of (model_type, model_size)
        """
        checkpoint_path = Path(checkpoint_path)

        # Check if it's a HuggingFace model (not a local path)
        if not checkpoint_path.exists() and "/" in str(checkpoint_path):
            # This is likely a HuggingFace model name
            try:
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(str(checkpoint_path), trust_remote_code=True)
                
                # Detect model type from config
                if hasattr(config, 'model_type'):
                    raw_model_type = config.model_type
                    print(f"DEBUG: Raw model type: {raw_model_type}")
                    # Map HuggingFace model types to our supported types
                    if raw_model_type in ['gpt2', 'gpt', 'gpt_neox']:
                        model_type = 'llama'  # Treat GPT models as Llama-like for conversion
                    elif raw_model_type in ['llama', 'mistral']:
                        model_type = 'llama'
                    elif raw_model_type in ['minicpm']:
                        model_type = 'minicpm'
                    else:
                        model_type = 'llama'  # Default for unknown (treat as Llama-like)
                    print(f"DEBUG: Mapped model type: {model_type}")
                elif hasattr(config, 'architectures') and config.architectures:
                    arch = config.architectures[0].lower()
                    if 'llama' in arch or 'mistral' in arch:
                        model_type = 'llama'
                    elif 'minicpm' in arch:
                        model_type = 'minicpm'
                    elif 'gpt2' in arch or 'gpt' in arch:
                        model_type = 'llama'  # Treat GPT models as Llama-like for conversion
                    else:
                        model_type = 'llama'  # Default for unknown (treat as Llama-like)
                else:
                    model_type = 'llama'  # Default (treat as Llama-like)
                
                # Detect model size from config
                if hasattr(config, 'num_hidden_layers'):
                    num_layers = config.num_hidden_layers
                    if num_layers == 32:
                        model_size = '8b'
                    elif num_layers == 24:
                        model_size = '3b'
                    elif num_layers == 40:
                        model_size = '14b'
                    elif num_layers == 18:
                        model_size = '1.5b'
                    elif num_layers == 12:
                        model_size = '0.5b'
                    else:
                        model_size = 'unknown'
                else:
                    model_size = 'unknown'
                
                return model_type, model_size
            except Exception as e:
                print(f"Warning: Could not detect HuggingFace model type: {e}")
                return 'minicpm', 'unknown'

        # Check if directory exists
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

        # Check for sharded files
        has_tp = any(checkpoint_path.glob("mp_rank_*"))

        if has_tp:
            # Has sharded files, indicating a large model
            # Try to load first shard to detect model
            first_ckpt = next(checkpoint_path.glob("mp_rank_*"))
            if (first_ckpt / "model_optim_rng.pt").exists():
                try:
                    state_dict = torch.load(first_ckpt / "model_optim_rng.pt", map_location="cpu")
                    if "model" in state_dict:
                        model_state = state_dict["model"]

                        # Determine model type by weight names
                        if "decoder.layers" in str(model_state.keys()):
                            # MiniCPM family
                            layer_count = len(
                                [k for k in model_state.keys() if "decoder.layers." in k and ".self_attention." in k]
                            )
                            if layer_count == 32:
                                return "minicpm", "8b"
                            elif layer_count == 24:
                                return "minicpm", "3b"
                            elif layer_count == 40:
                                return "minicpm", "14b"
                            elif layer_count == 18:
                                return "minicpm", "1.5b"
                            elif layer_count == 12:
                                return "minicpm", "0.5b"
                            else:
                                return "minicpm", "unknown"
                        elif "transformer.layers" in str(model_state.keys()):
                            # Llama family
                            layer_count = len([k for k in model_state.keys() if "transformer.layers." in k])
                            if layer_count == 32:
                                return "llama", "7b"
                            elif layer_count == 40:
                                return "llama", "13b"
                            elif layer_count == 60:
                                return "llama", "30b"
                            elif layer_count == 80:
                                return "llama", "65b"
                            else:
                                return "llama", "unknown"
                except Exception as e:
                    print(f"Warning: Could not load checkpoint for detection: {e}")

        # Default detection - conservative estimate
        return "minicpm", "3b"

    def detect_parallel_config(self, checkpoint_path: str) -> Dict[str, int]:
        """
        Detect parallel configuration from checkpoint structure

        Args:
            checkpoint_path: Path to the checkpoint directory

        Returns:
            Dictionary with tp_size and pp_size
        """
        checkpoint_path = Path(checkpoint_path)

        # Check TP size
        tp_ranks = list(checkpoint_path.glob("mp_rank_*"))
        tp_size = len(tp_ranks) if tp_ranks else 1

        # Check PP size (from filename format)
        pp_size = 1
        if tp_ranks:
            first_rank = tp_ranks[0]
            if "_" in first_rank.name:
                # Format: mp_rank_00_000 indicates PP
                pp_ranks = set([rank.name.split("_")[-1] for rank in tp_ranks])
                pp_size = len(pp_ranks)

        return {"tp_size": tp_size, "pp_size": pp_size}

    def detect_model_variant(self, checkpoint_path: str) -> str:
        """
        Detect if model is MiniCPM-4 (with MoE) or regular MiniCPM

        Args:
            checkpoint_path: Path to the checkpoint directory

        Returns:
            Model variant: "minicpm" or "minicpm4"
        """
        checkpoint_path = Path(checkpoint_path)

        # Check for MoE-related files in first shard
        first_ckpt = next(checkpoint_path.glob("mp_rank_*"))
        if (first_ckpt / "model_optim_rng.pt").exists():
            try:
                state_dict = torch.load(first_ckpt / "model_optim_rng.pt", map_location="cpu")
                if "model" in state_dict:
                    model_state = state_dict["model"]

                    # Check for MoE weights
                    has_moe = any("moe" in key.lower() for key in model_state.keys())
                    if has_moe:
                        return "minicpm4"
            except Exception:
                pass

        return "minicpm"

    def convert_megatron_to_hf(
        self,
        checkpoint_path: str,
        output_path: str,
        model_type: Optional[str] = None,
        model_size: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Smart conversion: Megatron -> HuggingFace

        Args:
            checkpoint_path: Input checkpoint directory
            output_path: Output HF weights path
            model_type: Model type (auto-detected if None)
            model_size: Model size (auto-detected if None)
            **kwargs: Additional conversion parameters
        """
        print(f"Starting smart conversion: {checkpoint_path} -> {output_path}")

        # 1. Auto-detect model type and size
        if model_type is None or model_size is None or model_type == "auto":
            detected_type, detected_size = self.detect_model_type_and_size(checkpoint_path)
            model_type = detected_type if model_type in (None, "auto") else model_type
            model_size = model_size or detected_size
            print(f"Auto-detection result: {model_type} {model_size}")

        # 2. Detect parallel configuration
        parallel_config = self.detect_parallel_config(checkpoint_path)
        print(f"Parallel config: TP={parallel_config['tp_size']}, PP={parallel_config['pp_size']}")

        # 3. Detect model variant for MiniCPM
        if model_type == "minicpm":
            model_variant = self.detect_model_variant(checkpoint_path)
            print(f"Model variant: {model_variant}")
        else:
            model_variant = model_type

        # 4. Get model configuration
        if model_type not in self.model_configs:
            raise ValueError(f"Unsupported model type: {model_type}")

        if model_size not in self.model_configs[model_type]:
            raise ValueError(f"Unsupported model size: {model_size}")

        config = self.model_configs[model_type][model_size]

        # 5. Select conversion strategy
        if config["use_basic"] and parallel_config["tp_size"] == 1:
            # Use basic conversion script
            print("Using basic conversion script...")
            if model_type == "minicpm":
                from .loader_minicpm_hf import load_checkpoint

                load_checkpoint(
                    None, type("Args", (), {"load": checkpoint_path, "save": output_path, **kwargs}), "megatron2hf"
                )
            elif model_type == "llama":
                from .loader_llama2 import load_checkpoint

                load_checkpoint(None, type("Args", (), {"load": checkpoint_path, "save": output_path, **kwargs}))
        else:
            # Use tensor parallel conversion script
            print("Using tensor parallel conversion script...")
            if model_variant == "minicpm4":
                # Use MiniCPM-4 specific converter
                from .dist_ckpt_to_hf_minicpm4 import convert_minicpm4_megatron_to_hf

                convert_minicpm4_megatron_to_hf(
                    checkpoint_path=checkpoint_path,
                    output_path=output_path,
                    num_layer=config["layers"],
                    tp_size=parallel_config["tp_size"],
                    pp_size=parallel_config["pp_size"],
                    num_kv_heads=config["num_kv_heads"],
                    num_query_heads=config["num_query_heads"],
                    **kwargs,
                )
            elif model_type == "minicpm":
                from .tp_pp_converter import TensorParallelConverter

                converter = TensorParallelConverter()
                converter.convert_minicpm_megatron_to_hf_tp_pp(
                    num_layer=config["layers"],
                    tp_size=parallel_config["tp_size"],
                    pp_size=parallel_config["pp_size"],
                    in_dir=checkpoint_path,
                    save_path=output_path,
                    num_kv_heads=config["num_kv_heads"],
                    num_query_heads=config["num_query_heads"],
                )
            elif model_type == "llama":
                from .tp_pp_converter import TensorParallelConverter

                converter = TensorParallelConverter()
                converter.convert_llama_megatron_to_hf_tp_pp(
                    num_layer=config["layers"],
                    tp_size=parallel_config["tp_size"],
                    pp_size=parallel_config["pp_size"],
                    in_dir=checkpoint_path,
                    save_path=output_path,
                    num_kv_heads=config["num_kv_heads"],
                    num_query_heads=config["num_query_heads"],
                )

        print(f"Conversion completed: {output_path}")

    def convert_hf_to_megatron(
        self,
        checkpoint_path: str,
        output_path: str,
        model_type: Optional[str] = None,
        model_size: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Smart conversion: HuggingFace -> Megatron

        Args:
            checkpoint_path: Input HF weights path
            output_path: Output Megatron checkpoint directory
            model_type: Model type (auto-detected if None)
            model_size: Model size (auto-detected if None)
            **kwargs: Additional conversion parameters
        """
        print(f"Starting smart conversion: {checkpoint_path} -> {output_path}")

        # Auto-detect if not provided
        if model_type is None or model_size is None or model_type == "auto":
            detected_type, detected_size = self.detect_model_type_and_size(checkpoint_path)
            model_type = detected_type if model_type in (None, "auto") else model_type
            model_size = model_size or detected_size
            print(f"Auto-detection result: {model_type} {model_size}")

        # Get model configuration
        if model_type not in self.model_configs:
            raise ValueError(f"Unsupported model type: {model_type}")

        if model_size not in self.model_configs[model_type]:
            raise ValueError(f"Unsupported model size: {model_size}")

        config = self.model_configs[model_type][model_size]

        # Use appropriate HF to Megatron converter
        if model_type == "minicpm":
            from .hf_to_megatron_minicpm import convert_hf_to_megatron_minicpm

            convert_hf_to_megatron_minicpm(
                checkpoint_path=checkpoint_path,
                output_path=output_path,
                num_layer=config["layers"],
                tp_size=config["tp_size"],
                pp_size=config["pp_size"],
                num_kv_heads=config["num_kv_heads"],
                num_query_heads=config["num_query_heads"],
                **kwargs,
            )
        elif model_type == "llama":
            from .loader_llama2_hf import load_checkpoint
            import queue
            import os

            # Create a queue for the conversion process
            conversion_queue = queue.Queue()
            
            # If it's a HuggingFace model name, download it first
            if "/" in checkpoint_path and not os.path.exists(checkpoint_path):
                from transformers import AutoModel
                print(f"Downloading model: {checkpoint_path}")
                model = AutoModel.from_pretrained(checkpoint_path, torch_dtype='auto', trust_remote_code=True)
                # Get the actual cached path
                from huggingface_hub import snapshot_download
                cached_path = snapshot_download(repo_id=checkpoint_path, local_files_only=True)
                checkpoint_path = cached_path
                print(f"Model downloaded to: {checkpoint_path}")
            
            # Create args object with required attributes
            megatron_path = os.path.join(os.path.dirname(__file__), "megatron")
            
            args_obj = type("Args", (), {
                "load": checkpoint_path,  # Use 'load' to match load_args_from_checkpoint
                "load_dir": checkpoint_path,  # Use 'load_dir' to match _load_checkpoint
                "save": output_path,
                "tokenizer_model": "tokenizer.model",  # Default tokenizer path
                "model_type": "GPT",  # Required for Llama models
                "megatron_path": megatron_path,  # Path to local Megatron
                **kwargs
            })
            
            load_checkpoint(conversion_queue, args_obj)

        print(f"Conversion completed: {output_path}")


# Convenience functions
def smart_convert_megatron_to_hf(checkpoint_path: str, output_path: str, **kwargs) -> None:
    """One-click smart conversion Megatron -> HF"""
    converter = SmartConverter()
    converter.convert_megatron_to_hf(checkpoint_path, output_path, **kwargs)


def smart_convert_hf_to_megatron(checkpoint_path: str, output_path: str, **kwargs) -> None:
    """One-click smart conversion HF -> Megatron"""
    converter = SmartConverter()
    converter.convert_hf_to_megatron(checkpoint_path, output_path, **kwargs)


# Model-specific convenience functions
def convert_minicpm_8b(checkpoint_path: str, output_path: str) -> None:
    """8B MiniCPM dedicated conversion"""
    converter = SmartConverter()
    converter.convert_megatron_to_hf(checkpoint_path, output_path, "minicpm", "8b")


def convert_minicpm_3b(checkpoint_path: str, output_path: str) -> None:
    """3B MiniCPM dedicated conversion"""
    converter = SmartConverter()
    converter.convert_megatron_to_hf(checkpoint_path, output_path, "minicpm", "3b")


def convert_llama_7b(checkpoint_path: str, output_path: str) -> None:
    """7B Llama dedicated conversion"""
    converter = SmartConverter()
    converter.convert_megatron_to_hf(checkpoint_path, output_path, "llama", "7b")


def convert_minicpm4_8b(checkpoint_path: str, output_path: str) -> None:
    """8B MiniCPM-4 dedicated conversion"""
    converter = SmartConverter()
    converter.convert_megatron_to_hf(checkpoint_path, output_path, "minicpm", "8b")
    