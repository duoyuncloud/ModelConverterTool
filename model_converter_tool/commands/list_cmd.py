import typer
from model_converter_tool.core.listing import list_supported

def list(
    target: str = typer.Argument("formats", help="What to list: formats or quantizations")
):
    """
    List supported formats or quantization types.
    """
    data = list_supported(target)
    if target == "formats":
        typer.echo("Supported input formats:")
        for k, v in data["input_formats"].items():
            typer.echo(f"  {k}: {v['description']}")
        typer.echo("\nSupported output formats:")
        for k, v in data["output_formats"].items():
            typer.echo(f"  {k}: {v['description']}")
    elif target == "quantizations":
        typer.echo("Supported quantization formats and options:")
        for k, opts in data.items():
            typer.echo(f"  {k}: {', '.join(opts) if opts else 'None'}")
    else:
        typer.echo(str(data)) 