import typer
from model_converter_tool.core.config import manage_config

def config(
    action: str = typer.Option("show", help="Action: show/get/set/list_presets"),
    key: str = typer.Option(None, help="Config key (for get/set)"),
    value: str = typer.Option(None, help="Config value (for set)")
):
    """
    Manage global/local configuration.
    """
    result = manage_config(action, key, value)
    if isinstance(result, dict):
        for k, v in result.items():
            typer.echo(f"{k}: {v}")
    else:
        typer.echo(str(result)) 