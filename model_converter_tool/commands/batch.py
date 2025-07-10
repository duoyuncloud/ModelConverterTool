import typer
import yaml
import json
import os
from pathlib import Path
from typing import List, Dict, Any
from model_converter_tool.core.conversion import convert_model
from model_converter_tool.utils import check_and_handle_disk_space, estimate_model_size, format_bytes

# Beautify parameter help
ARG_REQUIRED = "[bold red][required][/bold red]"
ARG_OPTIONAL = "[dim][optional][/dim]"


def batch(
    config_path: str = typer.Argument(..., help="Batch configuration file (YAML/JSON)"),
    max_workers: int = typer.Option(1, help="Maximum number of concurrent workers"),
    max_retries: int = typer.Option(1, help="Maximum number of retries per task"),
    skip_disk_check: bool = typer.Option(False, help="Skip disk space checking (not recommended)")
):
    """
    [dim]Examples:
      modelconvert batch configs/batch_template.yaml
      modelconvert batch my_batch_config.yaml --max-workers 2 --max-retries 3[/dim]

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
    
    # Load configuration file
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
    
    # Extract tasks from config
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
    
    # Check disk space for all tasks if not skipped
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
        
        # Check if we have enough space for all tasks
        if not check_and_handle_disk_space_batch(total_required_bytes):
            typer.echo("Batch conversion aborted due to insufficient disk space.")
            raise typer.Exit(1)
    
    # Execute batch conversion
    typer.echo(f"\n[bold]Starting batch conversion with {max_workers} worker(s)...[/bold]")
    
    results = []
    successful = 0
    failed = 0
    
    for i, task in enumerate(tasks):
        typer.echo(f"\n[bold]Task {i+1}/{len(tasks)}:[/bold] {task.get('model_path')} → {task.get('output_format')}")
        
        # Retry logic
        for retry in range(max_retries):
            try:
                result = convert_model(
                    input_path=task.get('model_path'),
                    output_path=task.get('output_path'),
                    to=task.get('output_format'),
                    quant=task.get('quantization'),
                    model_type=task.get('model_type', 'auto'),
                    device=task.get('device', 'auto'),
                    use_large_calibration=task.get('use_large_calibration', False)
                )
                
                if result.success:
                    typer.echo(f"  [green]✓[/green] Success: {result.output_path}")
                    successful += 1
                    break
                else:
                    typer.echo(f"  [red]✗[/red] Failed: {result.error}")
                    if retry < max_retries - 1:
                        typer.echo(f"  Retrying... (attempt {retry + 2}/{max_retries})")
                    else:
                        failed += 1
                        
            except Exception as e:
                typer.echo(f"  [red]✗[/red] Error: {str(e)}")
                if retry < max_retries - 1:
                    typer.echo(f"  Retrying... (attempt {retry + 2}/{max_retries})")
                else:
                    failed += 1
        
        results.append({
            'task': task,
            'success': result.success if 'result' in locals() else False,
            'output_path': result.output_path if 'result' in locals() and result.success else None,
            'error': result.error if 'result' in locals() and not result.success else str(e) if 'e' in locals() else None
        })
    
    # Summary
    typer.echo(f"\n[bold]Batch conversion completed:[/bold]")
    typer.echo(f"  Successful: {successful}")
    typer.echo(f"  Failed: {failed}")
    typer.echo(f"  Total: {len(tasks)}")
    
    if failed > 0:
        typer.echo(f"\n[yellow]Some tasks failed. Check the output above for details.[/yellow]")
        raise typer.Exit(1)
    else:
        typer.echo(f"\n[green]All tasks completed successfully![/green]")


def check_and_handle_disk_space_batch(total_required_bytes: int) -> bool:
    """
    Check disk space for batch operations.
    
    Args:
        total_required_bytes: Total required space in bytes
        
    Returns:
        True if operation should proceed, False if aborted
    """
    from model_converter_tool.utils import check_disk_space_safety, prompt_user_confirmation_low_space
    
    # Check disk space
    has_enough_space, space_info = check_disk_space_safety(
        total_required_bytes, safety_margin_gib=5.0, path="/"
    )
    
    if has_enough_space:
        # Log disk space info for transparency
        formatted = space_info["formatted"]
        typer.echo(f"[green]✓[/green] Sufficient disk space available:")
        typer.echo(f"  Free: {formatted['free']} | Required: {formatted['required']} | Safety margin: {formatted['safety_margin']}")
        return True
    
    # Check if we have enough space for the operation but not enough safety margin
    if space_info["has_enough_for_operation"] and not space_info["has_safety_margin"]:
        return prompt_user_confirmation_low_space(space_info)
    
    # Not enough space even for the operation
    formatted = space_info["formatted"]
    typer.echo(f"[red]❌[/red] INSUFFICIENT DISK SPACE")
    typer.echo(f"Current disk space:")
    typer.echo(f"  Free: {formatted['free']}")
    typer.echo(f"  Required: {formatted['required']}")
    typer.echo(f"  Shortage: {format_bytes(total_required_bytes - space_info['free_bytes'])}")
    typer.echo("Please free up disk space and try again.")
    return False 