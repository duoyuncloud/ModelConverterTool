import typer
from model_converter_tool.core.cache import manage_cache

def cache(action: str = typer.Option("show", help="Cache action (default: show)")):
    """
    Show local cache and workspace status.
    """
    result = manage_cache(action)
    if hasattr(result, 'workspace_path'):
        typer.echo(f"Workspace: {result.workspace_path}")
        typer.echo(f"Active tasks: {len(result.active_tasks)}")
        typer.echo(f"Completed: {len(result.completed_tasks)}")
        typer.echo(f"Failed: {len(result.failed_tasks)}")
    else:
        typer.echo(str(result)) 