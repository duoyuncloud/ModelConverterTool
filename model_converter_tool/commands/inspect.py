import typer
from model_converter_tool.core.inspect import inspect_model

def inspect(model: str):
    """
    Inspect and display detailed model information.
    """
    info = inspect_model(model)
    typer.echo(f"Path: {info.get('path')}")
    typer.echo(f"Format: {info.get('format')}")
    typer.echo(f"Detection confidence: {info.get('detection_confidence')}")
    if 'metadata' in info:
        typer.echo(f"Metadata: {info['metadata']}")
    if 'supported_outputs' in info:
        typer.echo(f"Convertible to: {', '.join(info['supported_outputs'])}")
    if 'error' in info:
        typer.echo(f"Error: {info['error']}") 