import typer
import yaml
import json
import os
from pathlib import Path
from typing import List, Dict, Any
from model_converter_tool.core.conversion import convert_model
from model_converter_tool.utils import check_and_handle_disk_space, estimate_model_size, format_bytes
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn, TaskProgressColumn

ARG_REQUIRED = "[bold red][required][/bold red]"
ARG_OPTIONAL = "[dim][optional][/dim]"

def batch(
    config_path: str = typer.Argument(..., help="Batch configuration file (YAML/JSON)"),
    max_workers: int = typer.Option(1, help="Maximum number of concurrent workers"),
    max_retries: int = typer.Option(1, help="Maximum number of retries per task"),
    skip_disk_check: bool = typer.Option(False, help="Skip disk space checking (not recommended)")
):
    """
    Batch convert multiple models using a configuration file.

    The configuration file should contain a list of conversion tasks, each with:
    - model_path: Input model path or repo id
    - output_format: Target format (onnx, gguf, torchscript, etc.)
    - output_path: Output file path
    - model_type: Model type (optional, default: auto)
    - device: Device (optional, default: auto)
    - quantization: Quantization type (optional)
    - use_large_calibration: Use large calibration dataset (optional, default: false)
    """
    config_file = Path(config_path)
    if not config_file.exists():
        typer.echo(f"[ERROR] Configuration file not found: {config_path}")
        raise typer.Exit(1)

    try:
        if config_file.suffix.lower() in ['.yaml', '.yml']:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        elif config_file.suffix.lower() == '.json':
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            typer.echo(f"[ERROR] Unsupported file format: {config_file.suffix}")
            typer.echo("Supported formats: .yaml, .yml, .json")
            raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"[ERROR] Failed to load configuration file: {e}")
        raise typer.Exit(1)

    if 'tasks' in config:
        tasks = config['tasks']
    elif 'models' in config:
        tasks = config['models']
    else:
        typer.echo("[ERROR] Configuration file must contain 'tasks' or 'models' key")
        raise typer.Exit(1)

    if not tasks:
        typer.echo("[ERROR] No tasks found in configuration file")
        raise typer.Exit(1)

    typer.echo(f"Loaded {len(tasks)} conversion tasks from {config_path}")

    if not skip_disk_check:
        typer.echo("\n[bold]Checking disk space for all tasks...[/bold]")
        total_required_bytes = 0
        for i, task in enumerate(tasks):
            model_path = task.get('model_path')
            output_format = task.get('output_format')
            quantization = task.get('quantization')
            if not model_path or not output_format:
                typer.echo(f"[ERROR] Task {i+1}: Missing required fields 'model_path' or 'output_format'")
                raise typer.Exit(1)
            estimated_size = estimate_model_size(model_path, output_format, quantization)
            total_required_bytes += estimated_size
            typer.echo(f"  Task {i+1}: {model_path} → {output_format} (estimated: {format_bytes(estimated_size)})")
        typer.echo(f"\nTotal estimated space required: {format_bytes(total_required_bytes)}")
        if not check_and_handle_disk_space_batch(total_required_bytes):
            typer.echo("Batch conversion aborted due to insufficient disk space.")
            raise typer.Exit(1)

    typer.echo(f"\n[bold]Starting batch conversion with {max_workers} worker(s)...[/bold]")

    results = []
    successful = 0
    failed = 0
    for i, task in enumerate(tasks):
        typer.echo(f"\n[bold]Task {i+1}/{len(tasks)}: {task.get('model_path')} → {task.get('output_format')}[/bold]")
        try:
            result = convert_model(
                input_path=task.get('model_path'),
                output_path=task.get('output_path'),
                to=task.get('output_format'),
                quant=task.get('quantization'),
                model_type=task.get('model_type', 'auto'),
                device=task.get('device', 'auto'),
                use_large_calibration=task.get('use_large_calibration', False),
                dtype=task.get('dtype'),
                quantization_config=task.get('quantization_config'),
                fake_weight=task.get('fake_weight', False)
            )
            if result.success:
                typer.echo(f"[green]Success! Output: {result.output_path}[/green]")
                successful += 1
            else:
                typer.echo(f"[red]Failed: {result.error}[/red]")
                failed += 1
            results.append(result)
        except Exception as e:
            typer.echo(f"[red]Exception during conversion: {e}[/red]")
            failed += 1
    typer.echo(f"\n[bold]Batch conversion completed: {successful} succeeded, {failed} failed.[/bold]")
    if failed > 0:
        raise typer.Exit(2)

def check_and_handle_disk_space_batch(total_required_bytes: int) -> bool:
    """
    Check disk space for batch operations.
    Returns True if operation should proceed, False if aborted.
    """
    from model_converter_tool.utils import check_disk_space_safety, prompt_user_confirmation_low_space
    has_enough_space, space_info = check_disk_space_safety(
        total_required_bytes, safety_margin_gib=5.0, path="/"
    )
    if has_enough_space:
        formatted = space_info["formatted"]
        typer.echo(f"[green]✓[/green] Sufficient disk space available:")
        typer.echo(f"  Free: {formatted['free']} | Required: {formatted['required']} | Safety margin: {formatted['safety_margin']}")
        return True
    if space_info["has_enough_for_operation"] and not space_info["has_safety_margin"]:
        return prompt_user_confirmation_low_space(space_info)
    formatted = space_info["formatted"]
    typer.echo(f"[red]❌[/red] INSUFFICIENT DISK SPACE")
    typer.echo(f"Current disk space:")
    typer.echo(f"  Free: {formatted['free']}")
    typer.echo(f"  Required: {formatted['required']}")
    typer.echo(f"  Shortage: {format_bytes(total_required_bytes - space_info['free_bytes'])}")
    typer.echo("Please free up disk space and try again.")
    return False 