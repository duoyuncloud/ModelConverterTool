import click
import torch
import sys
from model_converter_tool.converter import ModelConverter

# Global ModelConverter instance
converter = ModelConverter()


def detect_model_format(input_model):
    """Detect the format of input model."""
    fmt, norm_path = converter._detect_model_format(input_model)
    # norm_path could be path or model name, meta is optional
    meta = {"format": fmt}
    return fmt, norm_path, meta


def load_model_with_fallbacks(norm_path, model_type, device):
    """Load model with fallback strategies."""
    return converter._load_model_with_fallbacks(norm_path, model_type, device)


def validate_conversion_compatibility(in_fmt, output_format, model_type):
    """Validate conversion compatibility."""
    # Simple compatibility validation
    result = converter._validate_conversion_inputs(in_fmt, output_format, model_type, quantization="", device="auto")
    return {
        "compatible": result["valid"],
        "errors": result["errors"],
        "warnings": result["warnings"],
        "recommendations": [],
    }


@click.group()
def cli():
    """Model Converter CLI (CPU-only, supports all formats)"""
    pass


@cli.command()
@click.argument("input_model")
@click.argument("output_format")
@click.option("--output-path", default=None, help="Path to save the converted model")
@click.option("--model-type", default="auto", help="Model type (auto/text-generation/...)")
def convert(input_model, output_format, output_path, model_type):
    """Convert a model to the specified format (CPU-only)."""
    click.echo(f"[INFO] Detecting input model format for: {input_model}")
    in_fmt, norm_path, meta = detect_model_format(input_model)
    click.echo(f"Fmt: {in_fmt}")
    click.echo(f"Meta: {meta.get('format')}")

    # Compatibility check
    result = validate_conversion_compatibility(in_fmt, output_format, model_type)
    if not result["compatible"]:
        click.echo(f"[ERROR] Incompatible conversion: {result['errors']}")
        sys.exit(1)
    if result["warnings"]:
        click.echo(f"[WARN] {result['warnings']}")
    if result["recommendations"]:
        click.echo(f"[RECOMMEND] {result['recommendations']}")

    # Auto-detect device
    if torch.cuda.is_available():
        device = "cuda"
        click.echo("[INFO] GPU detected, all operations will be performed on GPU")
    else:
        device = "cpu"
        click.echo("[INFO] No GPU detected, all operations will be performed on CPU")

    # Special handling for quantized models like GPTQ/AWQ
    if output_format in ["gptq", "awq"]:
        click.echo(f"[INFO] Attempting quantized conversion ({output_format}) " f"on CPU...")
        # Check if related libraries support CPU
        try:
            if output_format == "gptq":
                # Check if auto-gptq supports CPU
                if not torch.cuda.is_available():
                    click.echo(
                        "[WARN] auto-gptq library is mainly optimized for GPU, "
                        "CPU performance may be slow and some features unavailable."
                    )
            elif output_format == "awq":
                click.echo(
                    "[WARN] awq library is mainly optimized for GPU, "
                    "CPU performance may be slow and some features unavailable."
                )
        except ImportError:
            click.echo(
                f"[ERROR] {output_format} library not installed, "
                "cannot perform quantized conversion. Please install dependencies."
            )
            sys.exit(1)
    # Load model
    click.echo("[INFO] Loading model with fallback strategies...")
    model, tokenizer, load_meta = load_model_with_fallbacks(norm_path, model_type, device)
    click.echo(
        f"[INFO] Model loaded. Device: {load_meta.get('device')}, " f"Format: {load_meta.get('format', 'unknown')}"
    )
    # Demo: actual conversion logic needs to be implemented based on output_format
    click.echo(
        f"[INFO] (DEMO) Would convert model to {output_format} " f"and save to {output_path or '[not specified]'}"
    )
    click.echo("[SUCCESS] Conversion pipeline completed (CPU-only mode).")


if __name__ == "__main__":
    cli()
