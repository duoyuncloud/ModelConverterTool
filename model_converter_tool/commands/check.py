import typer
from rich import print as rprint
from pathlib import Path
from model_converter_tool.core.check import check_model

def check(
    model_path: str = typer.Argument(..., help="Path to the model file to check."),
    model_format: str = typer.Option(None, "--format", "-f", help="Model format (e.g. gguf/onnx/mlx), can be auto-detected."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information."),
):
    """
    Check if a model file is usable (can be loaded and run a simple inference).
    """
    rprint(f"[bold cyan]Model usability check:[/bold cyan] {model_path}")

    result = check_model(model_path, model_format)
    if result["can_infer"]:
        rprint(f"[bold green]SUCCESS:[/bold green] Model can be loaded and run a simple inference.")
    else:
        rprint(f"[bold red]FAILED:[/bold red] Model could not be loaded or failed inference.")
        if result.get("error"):
            rprint(f"[red]Error: {result['error']}[/red]")
        if verbose and result.get("details"):
            rprint(f"[yellow]Details: {result['details']}[/yellow]")
        raise typer.Exit(2) 