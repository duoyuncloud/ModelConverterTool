import typer
from model_converter_tool.core.config import manage_config

ARG_OPTIONAL = "[dim][optional][/dim]"

def config(
    action: str = typer.Option("show", help=f"Action: show/get/set/list_presets. {ARG_OPTIONAL} Default: show"),
    key: str = typer.Option(None, help=f"Config key (for get/set). {ARG_OPTIONAL}"),
    value: str = typer.Option(None, help=f"Config value (for set). {ARG_OPTIONAL}")
):
    """
    Examples:
      modelconvert config show
      modelconvert config set cache_dir ./mycache

    Manage global/local configuration.\n
    Options:
      --action   Action: show/get/set/list_presets. [optional, default: show]
      --key      Config key (for get/set). [optional]
      --value    Config value (for set). [optional]
    """
    result = manage_config(action, key, value)
    if isinstance(result, dict):
        for k, v in result.items():
            typer.echo(f"{k}: {v}")
    else:
        typer.echo(str(result)) 