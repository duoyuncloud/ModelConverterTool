import typer
from model_converter_tool.core.history import get_history

def history():
    """
    [dim]Examples:
      modelconvert history
      modelconvert history  # (no arguments)[/dim]

    Show conversion history.
    """
    history = get_history()
    typer.echo("Completed tasks:")
    for t in history["completed"]:
        typer.echo(f"  {t}")
    typer.echo("Failed tasks:")
    for t in history["failed"]:
        typer.echo(f"  {t}")
    typer.echo("Active tasks:")
    for t in history["active"]:
        typer.echo(f"  {t}") 