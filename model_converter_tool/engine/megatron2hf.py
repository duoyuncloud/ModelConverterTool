"""
Megatron<->HF conversion engine for ModelConverterTool.
This engine provides unified stateless functions for both Megatron2HF and HF2Megatron directions,
using megatron_converters as the backend implementation.
"""

from megatron_converters import convert_llama, convert_minicpm


def convert_megatron_to_hf(model_type: str, checkpoint_path: str, output_path: str, **kwargs):
    """
    Convert a Megatron-format checkpoint to HuggingFace format.
    Args:
        model_type: 'llama' or 'minicpm'
        checkpoint_path: Path to Megatron checkpoint
        output_path: Path to save HuggingFace model
        kwargs: Additional arguments for the loader
    Returns:
        True if successful, False otherwise
    """

    class Args:
        def __init__(self, load_dir, save_dir, **kw):
            self.load_dir = load_dir
            self.save_dir = save_dir
            self.megatron_path = kw.get("megatron_path", None)
            self.tokenizer_model = kw.get("tokenizer_model", None)  # Add tokenizer_model attribute
            self.model_type = kw.get("model_type", None)  # Add model_type attribute
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
    print(
        f"[DEBUG] convert_hf_to_megatron called with model_type={model_type}, hf_path={hf_path}, output_path={output_path}, kwargs={kwargs}"
    )
    """
    Convert a HuggingFace-format checkpoint to Megatron format.
    Args:
        model_type: 'llama' or 'minicpm'
        hf_path: Path to HuggingFace model or model name
        output_path: Path to save Megatron checkpoint
        kwargs: Additional arguments for the loader
    Returns:
        True if successful, False otherwise
    """

    # Handle HuggingFace model names by downloading them first
    import os

    # Check if hf_path is a local path or a HuggingFace model name
    if os.path.exists(hf_path):
        local_model_path = hf_path
        print(f"[DEBUG] Using local model path: {local_model_path}")
    else:
        # Download the model to a temporary directory
        print(f"[DEBUG] Downloading model: {hf_path}")
        try:
            from transformers import AutoModelForCausalLM
            import tempfile

            # Create a temporary directory for the model
            temp_dir = tempfile.mkdtemp(prefix="megatron_convert_")
            print(f"[DEBUG] Temporary directory: {temp_dir}")

            # Download the model
            model = AutoModelForCausalLM.from_pretrained(hf_path, device_map="cpu", trust_remote_code=True)
            model.save_pretrained(temp_dir)

            # Also save the tokenizer if available
            try:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
                tokenizer.save_pretrained(temp_dir)
            except Exception as e:
                print(f"[DEBUG] Could not save tokenizer: {e}")

            local_model_path = temp_dir
            print(f"[DEBUG] Model downloaded to: {local_model_path}")
        except Exception as e:
            print(f"[DEBUG] Error downloading model: {e}")
            return False

    class Args:
        def __init__(self, load_dir, save_dir, **kw):
            self.load_dir = load_dir
            self.save_dir = save_dir
            self.megatron_path = kw.get("megatron_path", None)
            self.tokenizer_model = kw.get("tokenizer_model", None)  # Add tokenizer_model attribute
            self.model_type = kw.get("model_type", None)  # Add model_type attribute
            for k, v in kw.items():
                setattr(self, k, v)

    args = Args(load_dir=local_model_path, save_dir=output_path, model_type=model_type, **kwargs)
    print(
        f"[DEBUG] Args object created: load_dir={args.load_dir}, save_dir={args.save_dir}, model_type={args.model_type}"
    )
    if model_type == "llama":
        return convert_llama(args, direction="hf2megatron")
    elif model_type == "minicpm":
        return convert_minicpm(args, direction="hf2megatron")
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
