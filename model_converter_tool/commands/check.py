import typer
from rich import print as rprint
from model_converter_tool.core.check import check_model


def check(
    model_path: str = typer.Argument(..., help="Path to the model file to check."),
    model_format: str = typer.Option(
        None, "--format", "-f", help="Model format (e.g. gguf/onnx/mlx), can be auto-detected."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information."),
):
    """
    Check if a model file is usable (can be loaded and run a simple inference).
    """
    rprint(f"[bold cyan]Model usability check:[/bold cyan] {model_path}")

    result = check_model(model_path, model_format)
    if result["can_infer"]:
        rprint("[bold green]SUCCESS:[/bold green] Model can be loaded and run a simple inference.")
    else:
        rprint("[bold red]FAILED:[/bold red] Model could not be loaded or failed inference.")
        # 总是打印error/details，便于debug
        if result.get("error"):
            rprint(f"[red]Error: {result['error']}[/red]")
            # Enhanced: Show user-friendly message for common compatibility errors
            error_str = result["error"].lower()
            if "past_key_values" in error_str or "cache class" in error_str or "trust_remote_code" in error_str:
                rprint(
                    "[yellow]This error is likely due to model/Transformers version incompatibility, not a conversion failure. Check if the model requires a specific Transformers version or custom code.[/yellow]"
                )
        if result.get("details"):
            rprint(f"[yellow]Details: {result['details']}[/yellow]")
        if verbose:
            rprint(f"[yellow][VERBOSE] Full result: {result}[/yellow]")
        raise typer.Exit(2)
