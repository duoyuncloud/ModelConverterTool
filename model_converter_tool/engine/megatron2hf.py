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
        hf_path: Path to HuggingFace model
        output_path: Path to save Megatron checkpoint
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

    args = Args(load_dir=hf_path, save_dir=output_path, model_type=model_type, **kwargs)
    print(
        f"[DEBUG] Args object created: load_dir={args.load_dir}, save_dir={args.save_dir}, model_type={args.model_type}"
    )
    if model_type == "llama":
        return convert_llama(args, direction="hf2megatron")
    elif model_type == "minicpm":
        return convert_minicpm(args, direction="hf2megatron")
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
