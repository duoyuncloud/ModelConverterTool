"""
Megatron<->HF conversion engine for ModelConverterTool.
This engine provides unified stateless functions for both Megatron2HF and HF2Megatron directions,
using megatron_converters as the backend implementation.
"""

from megatron_converters import (
    # Legacy converters for backward compatibility
    convert_llama,
    convert_minicpm,
    # New tensor parallel converters
    SmartConverter,
    smart_convert_megatron_to_hf,
    smart_convert_hf_to_megatron,
    # Model-specific convenience functions
    convert_minicpm_8b,
    convert_minicpm_3b,
    convert_llama_7b,
    convert_minicpm_megatron_to_hf_tp_pp,
    convert_llama_megatron_to_hf_tp_pp,
    convert_hf_to_megatron_minicpm,
)


def convert_megatron_to_hf(model_type: str, checkpoint_path: str, output_path: str, **kwargs):
    """
    Convert a Megatron-format checkpoint to HuggingFace format.

    This function automatically detects the best conversion strategy:
    1. For tensor parallel models: Uses smart converter with auto-detection
    2. For basic models: Uses legacy converters
    3. For specific models: Uses model-specific converters

    Args:
        model_type: 'llama', 'minicpm', 'minicpm4', or 'auto' for auto-detection
        checkpoint_path: Path to Megatron checkpoint
        output_path: Path to save HuggingFace model
        kwargs: Additional arguments including:
            - tp_size: Tensor parallel size (auto-detected if not provided)
            - pp_size: Pipeline parallel size (auto-detected if not provided)
            - num_layer: Number of layers (auto-detected if not provided)
            - num_kv_heads: Number of KV heads (auto-detected if not provided)
            - num_query_heads: Number of query heads (auto-detected if not provided)
            - use_smart_converter: Force use of smart converter (default: True)
            - use_legacy_converter: Force use of legacy converter (default: False)

    Returns:
        True if successful, False otherwise
    """

    # Check if we should use smart converter (default behavior)
    use_smart_converter = kwargs.pop("use_smart_converter", True)
    use_legacy_converter = kwargs.pop("use_legacy_converter", False)

    # Force smart converter for auto detection
    if model_type == "auto":
        use_smart_converter = True
        use_legacy_converter = False

    if use_legacy_converter and model_type != "auto":
        # Use legacy converters for backward compatibility (but not for auto)
        return _convert_megatron_to_hf_legacy(model_type, checkpoint_path, output_path, **kwargs)

    if use_smart_converter or model_type == "auto":
        try:
            # Use smart converter with auto-detection
            smart_convert_megatron_to_hf(checkpoint_path, output_path, **kwargs)
            return True
        except Exception as e:
            print(f"Smart conversion failed: {e}")
            if model_type != "auto":
                print("Falling back to legacy converter...")
                return _convert_megatron_to_hf_legacy(model_type, checkpoint_path, output_path, **kwargs)
            else:
                print("Auto-detection failed and no fallback available for 'auto' model type")
                return False

    # Use model-specific converters if available
    if model_type == "minicpm":
        # Try to detect model size and use specific converter
        try:
            converter = SmartConverter()
            detected_type, detected_size = converter.detect_model_type_and_size(checkpoint_path)

            if detected_size == "8b":
                convert_minicpm_8b(checkpoint_path, output_path)
                return True
            elif detected_size == "3b":
                convert_minicpm_3b(checkpoint_path, output_path)
                return True
            else:
                # Use generic TP/PP converter
                parallel_config = converter.detect_parallel_config(checkpoint_path)
                model_config = converter.model_configs.get("minicpm", {}).get(detected_size, {})

                convert_minicpm_megatron_to_hf_tp_pp(
                    num_layer=model_config.get("layers", 24),
                    tp_size=parallel_config["tp_size"],
                    pp_size=parallel_config["pp_size"],
                    in_dir=checkpoint_path,
                    save_path=output_path,
                    num_kv_heads=model_config.get("num_kv_heads", 8),
                    num_query_heads=model_config.get("num_query_heads", 32),
                )
                return True
        except Exception as e:
            print(f"Model-specific conversion failed: {e}")
            return _convert_megatron_to_hf_legacy(model_type, checkpoint_path, output_path, **kwargs)

    elif model_type == "llama":
        try:
            converter = SmartConverter()
            detected_type, detected_size = converter.detect_model_type_and_size(checkpoint_path)

            if detected_size == "7b":
                convert_llama_7b(checkpoint_path, output_path)
                return True
            else:
                # Use generic TP/PP converter
                parallel_config = converter.detect_parallel_config(checkpoint_path)
                model_config = converter.model_configs.get("llama", {}).get(detected_size, {})

                convert_llama_megatron_to_hf_tp_pp(
                    num_layer=model_config.get("layers", 32),
                    tp_size=parallel_config["tp_size"],
                    pp_size=parallel_config["pp_size"],
                    in_dir=checkpoint_path,
                    save_path=output_path,
                    num_kv_heads=model_config.get("num_kv_heads", 8),
                    num_query_heads=model_config.get("num_query_heads", 32),
                )
                return True
        except Exception as e:
            print(f"Model-specific conversion failed: {e}")
            return _convert_megatron_to_hf_legacy(model_type, checkpoint_path, output_path, **kwargs)

    else:
        # Fall back to legacy converter
        return _convert_megatron_to_hf_legacy(model_type, checkpoint_path, output_path, **kwargs)


def _convert_megatron_to_hf_legacy(model_type: str, checkpoint_path: str, output_path: str, **kwargs):
    """
    Legacy conversion function for backward compatibility.
    """

    class Args:
        def __init__(self, load_dir, save_dir, **kw):
            self.load_dir = load_dir
            self.save_dir = save_dir
            self.megatron_path = kw.get("megatron_path", None)
            self.tokenizer_model = kw.get("tokenizer_model", None)
            self.model_type = kw.get("model_type", None)
            self.checkpoint_type = kw.get("checkpoint_type", "hf")
            self.model_size = kw.get("model_size", "7B")
            self.bf16 = kw.get("bf16", False)
            self.fp16 = kw.get("fp16", False)
            self.true_vocab_size = kw.get("true_vocab_size", None)
            self.vocab_file = kw.get("vocab_file", None)
            self.loader_transformer_impl = kw.get("loader_transformer_impl", "local")
            for k, v in kw.items():
                setattr(self, k, v)

    args = Args(load_dir=checkpoint_path, save_dir=output_path, model_type=model_type, **kwargs)
    if model_type == "llama":
        return convert_llama(args, direction="megatron2hf")
    elif model_type == "minicpm":
        return convert_minicpm(args, direction="megatron2hf")
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def convert_hf_to_megatron(model_type: str, hf_path: str, output_path: str, **kwargs):
    """
    Convert a HuggingFace-format checkpoint to Megatron format.

    This function automatically detects the best conversion strategy:
    1. For tensor parallel models: Uses smart converter with auto-detection
    2. For basic models: Uses legacy converters
    3. For specific models: Uses model-specific converters

    Args:
        model_type: 'llama', 'minicpm', 'minicpm4', or 'auto' for auto-detection
        hf_path: Path to HuggingFace model or model name
        output_path: Path to save Megatron checkpoint
        kwargs: Additional arguments including:
            - tp_size: Tensor parallel size
            - pp_size: Pipeline parallel size
            - num_layer: Number of layers
            - num_kv_heads: Number of KV heads
            - num_query_heads: Number of query heads
            - use_smart_converter: Force use of smart converter (default: True)
            - use_legacy_converter: Force use of legacy converter (default: False)

    Returns:
        True if successful, False otherwise
    """

    # Check if we should use smart converter (default behavior)
    use_smart_converter = kwargs.pop("use_smart_converter", True)
    use_legacy_converter = kwargs.pop("use_legacy_converter", False)

    # Force smart converter for auto detection
    if model_type == "auto":
        use_smart_converter = True
        use_legacy_converter = False

    if use_legacy_converter and model_type != "auto":
        # Use legacy converters for backward compatibility (but not for auto)
        return _convert_hf_to_megatron_legacy(model_type, hf_path, output_path, **kwargs)

    if use_smart_converter or model_type == "auto":
        try:
            # Use smart converter with auto-detection
            smart_convert_hf_to_megatron(hf_path, output_path, **kwargs)
            return True
        except Exception as e:
            print(f"Smart conversion failed: {e}")
            if model_type != "auto":
                print("Falling back to legacy converter...")
                return _convert_hf_to_megatron_legacy(model_type, hf_path, output_path, **kwargs)
            else:
                print("Auto-detection failed and no fallback available for 'auto' model type")
                return False

    # Use model-specific converters if available
    if model_type == "minicpm":
        try:
            # Use HF to Megatron converter
            converter = SmartConverter()
            detected_type, detected_size = converter.detect_model_type_and_size(hf_path)

            model_config = converter.model_configs.get("minicpm", {}).get(detected_size, {})

            convert_hf_to_megatron_minicpm(
                checkpoint_path=hf_path,
                output_path=output_path,
                num_layer=model_config.get("layers", 24),
                tp_size=kwargs.get("tp_size", model_config.get("tp_size", 1)),
                pp_size=kwargs.get("pp_size", model_config.get("pp_size", 1)),
                num_kv_heads=kwargs.get("num_kv_heads", model_config.get("num_kv_heads", 8)),
                num_query_heads=kwargs.get("num_query_heads", model_config.get("num_query_heads", 32)),
            )
            return True
        except Exception as e:
            print(f"Model-specific conversion failed: {e}")
            return _convert_hf_to_megatron_legacy(model_type, hf_path, output_path, **kwargs)

    elif model_type == "llama":
        try:
            # Use HF to Megatron converter
            converter = SmartConverter()
            detected_type, detected_size = converter.detect_model_type_and_size(hf_path)

            model_config = converter.model_configs.get("llama", {}).get(detected_size, {})

            # For Llama, we'll use the legacy converter as it's more mature
            return _convert_hf_to_megatron_legacy(model_type, hf_path, output_path, **kwargs)
        except Exception as e:
            print(f"Model-specific conversion failed: {e}")
            return _convert_hf_to_megatron_legacy(model_type, hf_path, output_path, **kwargs)

    else:
        # Fall back to legacy converter
        return _convert_hf_to_megatron_legacy(model_type, hf_path, output_path, **kwargs)


def _convert_hf_to_megatron_legacy(model_type: str, hf_path: str, output_path: str, **kwargs):
    """
    Legacy conversion function for backward compatibility.
    """
    # Handle HuggingFace model names by downloading them first
    import os

    # Check if hf_path is a local path or a HuggingFace model name
    if os.path.exists(hf_path):
        local_model_path = hf_path
    else:
        # Download the model to a temporary directory
        try:
            from transformers import AutoModelForCausalLM
            import tempfile

            # Create a temporary directory for the model
            temp_dir = tempfile.mkdtemp(prefix="megatron_convert_")

            # Download the model
            model = AutoModelForCausalLM.from_pretrained(hf_path, device_map="cpu", trust_remote_code=True)
            model.save_pretrained(temp_dir)

            # Also save the tokenizer if available
            try:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
                tokenizer.save_pretrained(temp_dir)
            except Exception:
                pass

            local_model_path = temp_dir
        except Exception:
            return False

    class Args:
        def __init__(self, load_dir, save_dir, **kw):
            self.load_dir = load_dir
            self.save_dir = save_dir
            self.megatron_path = kw.get("megatron_path", None)
            self.tokenizer_model = kw.get("tokenizer_model", None)
            self.model_type = kw.get("model_type", None)
            self.checkpoint_type = kw.get("checkpoint_type", "hf")
            self.model_size = kw.get("model_size", "7B")
            self.bf16 = kw.get("bf16", False)
            self.fp16 = kw.get("fp16", False)
            self.true_vocab_size = kw.get("true_vocab_size", None)
            self.vocab_file = kw.get("vocab_file", None)
            self.loader_transformer_impl = kw.get("loader_transformer_impl", "local")
            for k, v in kw.items():
                setattr(self, k, v)

    args = Args(load_dir=local_model_path, save_dir=output_path, model_type=model_type, **kwargs)
    if model_type == "llama":
        return convert_llama(args, direction="hf2megatron")
    elif model_type == "minicpm":
        return convert_minicpm(args, direction="hf2megatron")
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def detect_model_type_and_size(checkpoint_path: str) -> tuple[str, str]:
    """
    Detect model type and size from checkpoint path.

    Args:
        checkpoint_path: Path to the checkpoint directory or file

    Returns:
        Tuple of (model_type, model_size)
    """
    try:
        converter = SmartConverter()
        return converter.detect_model_type_and_size(checkpoint_path)
    except Exception as e:
        print(f"Auto-detection failed: {e}")
        return "minicpm", "3b"  # Default fallback


def detect_parallel_config(checkpoint_path: str) -> dict:
    """
    Detect parallel configuration from checkpoint path.

    Args:
        checkpoint_path: Path to the checkpoint directory

    Returns:
        Dictionary with tp_size and pp_size
    """
    try:
        converter = SmartConverter()
        return converter.detect_parallel_config(checkpoint_path)
    except Exception as e:
        print(f"Parallel config detection failed: {e}")
        return {"tp_size": 1, "pp_size": 1}  # Default fallback
