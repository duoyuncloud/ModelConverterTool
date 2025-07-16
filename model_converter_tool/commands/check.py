import typer
from rich import print as rprint
from pathlib import Path
import importlib

from model_converter_tool.converter import ModelConverter

# Supported formats for usability check
SUPPORTED_FORMATS = [
    "gguf", "onnx", "mlx", "gptq", "awq", "safetensors", "torchscript", "hf"
]

# Map format to (engine module, check function)
FORMAT_CHECKS = {
    "gguf": ("model_converter_tool.engine.gguf", "validate_gguf_file"),
    "onnx": ("model_converter_tool.engine.onnx", "validate_onnx_file"),
    "mlx": ("model_converter_tool.engine.mlx", "validate_mlx_file"),
    "gptq": ("model_converter_tool.engine.gptq", "validate_gptq_file"),
    "awq": ("model_converter_tool.engine.awq", "validate_awq_file"),
    "safetensors": ("model_converter_tool.engine.safetensors", "validate_safetensors_file"),
    "torchscript": ("model_converter_tool.engine.torchscript", "validate_torchscript_file"),
    "hf": ("model_converter_tool.engine.hf", "validate_hf_file"),
}

def check(
    model_path: str = typer.Argument(..., help="Path to the model file to check."),
    model_format: str = typer.Option(None, "--format", "-f", help="Model format (e.g. gguf/onnx/mlx), can be auto-detected."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information."),
):
    """
    Check if a model file is usable (can be loaded and run a simple inference).
    """
    rprint(f"[bold cyan]Model usability check:[/bold cyan] {model_path}")

    # Auto-detect format if not provided
    if not model_format:
        converter = ModelConverter()
        detected_format, norm_path = converter._detect_model_format(model_path)
        model_format = detected_format
        model_path = norm_path
        rprint(f"Detected format: [green]{model_format}[/green]")
    else:
        model_format = model_format.lower()

    if model_format == "huggingface":
        model_format = "hf"
    elif model_format == "hf":
        model_format = "hf"

    if model_format not in SUPPORTED_FORMATS:
        rprint(f"[red]Format '{model_format}' is not supported for usability check. Supported: {SUPPORTED_FORMATS}[/red]")
        raise typer.Exit(1)

    engine_module, check_func_name = FORMAT_CHECKS[model_format]
    try:
        engine = importlib.import_module(engine_module)
        check_func = getattr(engine, check_func_name)
        # Always pass two arguments for compatibility
        arg = Path(model_path) if model_format in ("gguf", "mlx") else model_path
        try:
            result = check_func(arg, None)
        except Exception as check_exc:
            rprint(f"[bold red]EXCEPTION in check_func:[/bold red] {check_exc}")
            if verbose:
                import traceback
                traceback.print_exc()
            raise
        if result:
            rprint(f"[bold green]SUCCESS:[/bold green] Model can be loaded and run a simple inference.")
        else:
            rprint(f"[bold red]FAILED:[/bold red] Model could not be loaded or failed inference.")
            raise typer.Exit(2)
    except Exception as e:
        rprint(f"[bold red]ERROR:[/bold red] Exception during check: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        raise typer.Exit(3) 