#!/usr/bin/env python3
"""
Smart Conversion Scheduler
Automatically selects the best conversion script based on model type, size, and parallel strategy
"""

import torch
import os
from typing import Dict, Optional, Tuple
from pathlib import Path


class SmartConverter:
    """Smart conversion scheduler for automatic model conversion"""

    def __init__(self):
        self.model_configs = {
            # MiniCPM configurations
            "minicpm": {
                "8b": {
                    "num_layer": 32,
                    "tp_size": 2,
                    "pp_size": 1,
                    "num_kv_heads": 8,
                    "num_query_heads": 32,
                },
                "4b": {
                    "num_layer": 62,
                    "tp_size": 2,
                    "pp_size": 1,
                    "num_kv_heads": 8,
                    "num_query_heads": 32,
                },
                "3b": {
                    "num_layer": 24,
                    "tp_size": 1,
                    "pp_size": 1,
                    "num_kv_heads": 8,
                    "num_query_heads": 32,
                },
            },
            # Llama configurations
            "llama": {
                "7b": {
                    "num_layer": 32,
                    "tp_size": 1,
                    "pp_size": 1,
                    "num_kv_heads": 8,
                    "num_query_heads": 32,
                },
                "13b": {
                    "num_layer": 40,
                    "tp_size": 2,
                    "pp_size": 1,
                    "num_kv_heads": 8,
                    "num_query_heads": 40,
                },
                "30b": {
                    "num_layer": 60,
                    "tp_size": 4,
                    "pp_size": 1,
                    "num_kv_heads": 8,
                    "num_query_heads": 52,
                },
                "65b": {
                    "num_layer": 80,
                    "tp_size": 8,
                    "pp_size": 1,
                    "num_kv_heads": 8,
                    "num_query_heads": 64,
                },
                "0.5b": {
                    "num_layer": 12,
                    "tp_size": 1,
                    "pp_size": 1,
                    "num_kv_heads": 4,
                    "num_query_heads": 16,
                },
                "1.1b": {
                    "num_layer": 22,
                    "tp_size": 1,
                    "pp_size": 1,
                    "num_kv_heads": 4,
                    "num_query_heads": 32,
                },
            },
        }

    def detect_model_type_and_size(self, config) -> Tuple[str, str]:
        """
        Detect model type and size from HuggingFace config
        """
        print(f"DEBUG: Raw model type: {config.model_type}")
        
        # Detect model type from config
        if hasattr(config, 'model_type'):
            raw_model_type = config.model_type
            # Map HuggingFace model types to our supported types
            if raw_model_type in ['gpt2', 'gpt', 'gpt_neox']:
                model_type = 'llama'  # Treat GPT models as Llama-like for conversion
            elif raw_model_type in ['llama', 'mistral']:
                model_type = 'llama'
            elif raw_model_type in ['minicpm', 'minicpm3']: # Added 'minicpm3'
                model_type = 'minicpm'
            else:
                model_type = 'llama'  # Default for unknown (treat as Llama-like)
        else:
            model_type = 'llama'  # Default fallback
        
        print(f"DEBUG: Mapped model type: {model_type}")
        
        # Detect model size based on parameters
        if hasattr(config, 'num_hidden_layers') and hasattr(config, 'hidden_size'):
            num_layers = config.num_hidden_layers
            hidden_size = config.hidden_size
            
            # Calculate approximate model size in billions
            vocab_size = getattr(config, 'vocab_size', 32000)
            num_heads = getattr(config, 'num_attention_heads', 32)
            
            # Rough parameter count estimation
            # Embedding: vocab_size * hidden_size
            # Each layer: 4 * hidden_size * hidden_size (QKV + output + 2 MLP layers)
            # Final norm: hidden_size
            # Output layer: vocab_size * hidden_size
            
            embedding_params = vocab_size * hidden_size
            layer_params = num_layers * (4 * hidden_size * hidden_size + hidden_size)
            output_params = vocab_size * hidden_size
            total_params = embedding_params + layer_params + output_params
            
            # Convert to billions
            model_size_b = total_params / 1e9
            
            print(f"DEBUG: Estimated model size: {model_size_b:.1f}B parameters")
            
            # Map to closest size
            if model_type == 'llama':
                if model_size_b < 0.8:
                    size = '0.5b'
                elif model_size_b < 1.5:
                    size = '1.1b'
                elif model_size_b < 10:
                    size = '7b'
                elif model_size_b < 20:
                    size = '13b'
                elif model_size_b < 40:
                    size = '30b'
                else:
                    size = '65b'
            elif model_type == 'minicpm':
                if model_size_b < 2:
                    size = '3b'
                elif model_size_b < 6:
                    size = '4b'
                else:
                    size = '8b'
            else:
                size = 'unknown'
        else:
            size = 'unknown'
        
        print(f"Auto-detection result: {model_type} {size}")
        return model_type, size

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

            # Detect if model uses MLA by checking the original model type
            use_mla = False
            if "/" in checkpoint_path and not os.path.exists(checkpoint_path):
                # This is a HuggingFace model name, check the original model type
                try:
                    from transformers import AutoConfig
                    config_obj = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)
                    if hasattr(config_obj, 'model_type') and config_obj.model_type == 'minicpm3':
                        use_mla = True
                        print(f"Detected MiniCPM3 model, using MLA attention")
                except Exception as e:
                    print(f"Warning: Could not detect MLA usage: {e}")

            convert_hf_to_megatron_minicpm(
                checkpoint_path=checkpoint_path,
                output_path=output_path,
                num_layer=config["layers"],
                tp_size=config["tp_size"],
                pp_size=config["pp_size"],
                num_kv_heads=config["num_kv_heads"],
                num_query_heads=config["num_query_heads"],
                use_mla=use_mla,
                **kwargs,
            )
        elif model_type == "llama":
            from .loader_llama2_hf import load_checkpoint
            import queue

            # Create a queue for the conversion process
            conversion_queue = queue.Queue()
            
            # If it's a HuggingFace model name, download it first
            if "/" in checkpoint_path and not os.path.exists(checkpoint_path):
                from transformers import AutoModelForCausalLM
                print(f"Downloading model: {checkpoint_path}")
                model = AutoModelForCausalLM.from_pretrained(checkpoint_path, torch_dtype='auto', trust_remote_code=True)
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

    def smart_convert(self, checkpoint_path: str, output_path: str, model_type: str = None) -> bool:
        """
        Smart conversion that automatically detects model type and configuration
        """
        print(f"Starting smart conversion: {checkpoint_path} -> {output_path}")
        
        try:
            # Load config for detection
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)
            
            # Auto-detect if not specified
            if model_type is None or model_type == "auto":
                detected_type, detected_size = self.detect_model_type_and_size(config)
            else:
                detected_type = model_type
                detected_size = "unknown"
            
            # Get configuration
            if detected_type in self.model_configs and detected_size in self.model_configs[detected_type]:
                config_dict = self.model_configs[detected_type][detected_size]
                print(f"Using configuration: {detected_type} {detected_size}")
                print(f"Config: {config_dict}")
                
                # Perform conversion
                if detected_type == "minicpm":
                    from .hf_to_megatron_minicpm import convert_hf_to_megatron_minicpm
                    convert_hf_to_megatron_minicpm(
                        checkpoint_path=checkpoint_path,
                        output_path=output_path,
                        num_layer=config_dict["num_layer"],
                        tp_size=config_dict["tp_size"],
                        pp_size=config_dict["pp_size"],
                        num_kv_heads=config_dict["num_kv_heads"],
                        num_query_heads=config_dict["num_query_heads"],
                    )
                    return True
                elif detected_type == "llama":
                    # Use the legacy converter for Llama models
                    print("Using legacy converter for Llama models")
                    return False
                else:
                    print(f"Unsupported model type: {detected_type}")
                    return False
            else:
                print(f"Smart conversion failed: Unsupported model size: {detected_size}")
                return False
                
        except Exception as e:
            print(f"Smart conversion failed: {e}")
            return False


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