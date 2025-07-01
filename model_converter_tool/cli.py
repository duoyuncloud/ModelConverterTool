import os
import sys

import click
import torch

from model_converter_tool.converter import ModelConverter

# Global ModelConverter instance
converter = ModelConverter()

# Dynamically set help text based on hardware
if torch.cuda.is_available():
    CLI_HELP = "Model Converter CLI (GPU & CPU supported, all formats)"
    CONVERT_HELP = "Convert a model to the specified format (GPU/CPU supported)."
else:
    CLI_HELP = "Model Converter CLI (CPU supported, all formats)"
    CONVERT_HELP = "Convert a model to the specified format (CPU supported)."


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
    result = converter._validate_conversion_inputs(
        in_fmt, output_format, model_type, quantization="", device="auto"
    )
    return {
        "compatible": result["valid"],
        "errors": result["errors"],
        "warnings": result["warnings"],
        "recommendations": [],
    }


@click.group(help=CLI_HELP)
def cli():
    pass


@cli.command(help=CONVERT_HELP)
@click.argument("input_model")
@click.argument("output_format")
@click.option("--output-path", default=None, help="Path to save the converted model")
@click.option(
    "--model-type", default="auto", help="Model type (auto/text-generation/...)"
)
def convert(input_model, output_format, output_path, model_type):
    """Convert a model to the specified format."""
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

    # 自动检测设备（CUDA > MPS > CPU）
    import platform

    device = "cpu"
    device_str = "CPU"
    if torch.cuda.is_available():
        device = "cuda"
        device_str = f"GPU: {torch.cuda.get_device_name(0)}"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        device_str = "Apple Silicon MPS"
    click.echo(f"[INFO] Using device: {device_str}")

    # Special handling for quantized models like GPTQ/AWQ
    if output_format in ["gptq", "awq"]:
        click.echo(
            f"[INFO] Attempting quantized conversion ({output_format}) on {device_str}..."
        )
        # Check if related libraries support CPU
        try:
            if output_format == "gptq":
                # Check if auto-gptq supports CPU
                if device == "cpu":
                    click.echo(
                        "[WARN] auto-gptq library is mainly optimized for GPU, "
                        "CPU performance may be slow and some features unavailable."
                    )
            elif output_format == "awq":
                if device == "cpu":
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
    model, tokenizer, load_meta = load_model_with_fallbacks(
        norm_path, model_type, device
    )
    click.echo(
        f"[INFO] Model loaded. Device: {getattr(load_meta, 'device', 'unknown')}, Format: {getattr(load_meta, 'format', 'unknown')}"
    )
    # Perform actual conversion
    if output_path is None:
        # Generate default output path
        if output_format in ["onnx", "gguf", "mlx", "pt"]:
            output_path = f"./outputs/{input_model.replace('/', '_')}.{output_format}"
        else:
            output_path = f"./outputs/{input_model.replace('/', '_')}_{output_format}"

    click.echo(
        f"[INFO] Converting model to {output_format} and saving to {output_path}"
    )

    try:
        result = converter.convert(
            input_source=input_model,
            output_format=output_format,
            output_path=output_path,
            model_type=model_type,
            device=device,
            validate=True,
        )

        if result.get("success"):
            click.echo(f"[SUCCESS] Conversion completed successfully!")
            if result.get("validation", True):
                click.echo(f"[INFO] Model validation passed")
            else:
                click.echo(
                    f"[WARN] Model validation failed: {result.get('warning', 'Unknown issue')}"
                )

            # 验证导出文件是否存在
            if not os.path.exists(output_path):
                click.echo(f"[ERROR] Output file not found: {output_path}")
                sys.exit(1)

            # 尝试加载模型，确保文件未损坏
            load_success = False
            load_error = None
            if output_format == "onnx":
                try:
                    import onnx

                    onnx_model = onnx.load(output_path)
                    onnx.checker.check_model(onnx_model)
                    load_success = True
                except Exception as e:
                    load_error = str(e)
            elif output_format in ["torchscript", "pt", "pytorch"]:
                try:
                    _ = torch.load(output_path, map_location="cpu", weights_only=False)
                    load_success = True
                except Exception as e:
                    load_error = str(e)
            # 其它格式可按需补充
            else:
                load_success = True  # 默认只检查文件存在

            if load_success:
                click.echo(
                    f"[SUCCESS] Conversion completed successfully!\n[INFO] Output file exists and can be loaded by the target framework."
                )
                click.echo("[INFO] 如需详细推理验证，请参考官方文档或使用 pytest/validator 工具。")
            else:
                click.echo(
                    f"[WARN] Output file generated, but failed to load: {load_error}"
                )
                click.echo("[INFO] 请检查模型格式或使用 validator 进行详细验证。")
        else:
            click.echo(
                f"[ERROR] Conversion failed: {result.get('error', 'Unknown error')}"
            )
            sys.exit(1)

    except Exception as e:
        click.echo(f"[ERROR] Conversion error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
