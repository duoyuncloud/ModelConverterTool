import typer
from model_converter_tool.core.config import manage_config

ARG_OPTIONAL = "[dim][optional][/dim]"

app = typer.Typer(help="Manage global/local configuration.")


@app.command()
def show():
    """Show all configuration values."""
    result = manage_config("show", None, None)
    if isinstance(result, dict):
        for k, v in result.items():
            typer.echo(f"{k}: {v}")
    else:
        typer.echo(str(result))


@app.command()
def get(key: str = typer.Argument(..., help="Config key to get.")):
    """Get a configuration value by key."""
    result = manage_config("get", key, None)
    typer.echo(str(result))


@app.command()
def set(
    key: str = typer.Argument(..., help="Config key to set."), value: str = typer.Argument(..., help="Value to set.")
):
    """Set a configuration value."""
    result = manage_config("set", key, value)
    typer.echo(str(result))


@app.command("list-presets")
def list_presets():
    """List all available configuration presets."""
    result = manage_config("list_presets", None, None)
    typer.echo(str(result))
