import typer
import yaml
import json
import os
from pathlib import Path
from typing import List, Dict, Any
from model_converter_tool.core.convert import convert_model
from model_converter_tool.utils import check_and_handle_disk_space, estimate_model_size, format_bytes
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn, TaskProgressColumn
from rich.console import Console

ARG_REQUIRED = "[bold red][required][/bold red]"
ARG_OPTIONAL = "[dim][optional][/dim]"

def auto_complete_output_path(input_path, output_path, to_format):
    import os
    from pathlib import Path
    output_aliases = {"hf": "huggingface"}
    to_format = output_aliases.get(to_format.lower(), to_format.lower())
    file_exts = {
        'onnx': '.onnx',
        'gguf': '.gguf',
        'pt': '.pt',
        'torchscript': '.pt',
        'safetensors': '.safetensors',
        'fp16': '.safetensors',
    }
    base = os.path.splitext(os.path.basename(input_path))[0]
    def to_dir_name(path, ext=None):
        p = Path(path)
        if ext and p.name.endswith(ext):
            return str(p.with_suffix('')) + f'_{to_format}'
        if p.suffix:
            return str(p.with_suffix('')) + f'_{to_format}'
        return str(p)
    if not output_path:
        return f'./outputs/{base}_{to_format}'
    if os.path.isdir(output_path):
        return output_path
    for ext in file_exts.values():
        if output_path.endswith(ext):
            return to_dir_name(output_path, ext)
    if not os.path.exists(output_path) and not output_path.endswith('/'):
        return to_dir_name(output_path)
    return output_path

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
    console = Console()
    config_file = Path(config_path)
    if not config_file.exists():
        console.print(f"[ERROR] Configuration file not found: {config_path}")
        raise typer.Exit(1)

    try:
        if config_file.suffix.lower() in ['.yaml', '.yml']:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        elif config_file.suffix.lower() == '.json':
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            console.print(f"[ERROR] Unsupported file format: {config_file.suffix}")
            console.print("Supported formats: .yaml, .yml, .json")
            raise typer.Exit(1)
    except Exception as e:
        console.print(f"[ERROR] Failed to load configuration file: {e}")
        raise typer.Exit(1)

    if 'tasks' in config:
        tasks = config['tasks']
    elif 'models' in config:
        tasks = config['models']
    else:
        console.print("[ERROR] Configuration file must contain 'tasks' or 'models' key")
        raise typer.Exit(1)

    if not tasks:
        console.print("[ERROR] No tasks found in configuration file")
        raise typer.Exit(1)

    console.print(f"Loaded {len(tasks)} conversion tasks from {config_path}")

    if not skip_disk_check:
        console.print("\n[bold]Checking disk space for all tasks...[/bold]")
        total_required_bytes = 0
        for i, task in enumerate(tasks):
            model_path = task.get('model_path')
            output_format = task.get('output_format')
            quantization = task.get('quantization')
            if not model_path or not output_format:
                console.print(f"[ERROR] Task {i+1}: Missing required fields 'model_path' or 'output_format'")
                raise typer.Exit(1)
            estimated_size = estimate_model_size(model_path, output_format, quantization)
            total_required_bytes += estimated_size
            console.print(f"  Task {i+1}: {model_path} → {output_format} (estimated: {format_bytes(estimated_size)})")
        console.print(f"\nTotal estimated space required: {format_bytes(total_required_bytes)}")
        if not check_and_handle_disk_space_batch(total_required_bytes, console):
            console.print("Batch conversion aborted due to insufficient disk space.")
            raise typer.Exit(1)

    console.print(f"\n[bold]Starting batch conversion with {max_workers} worker(s)...[/bold]")

    results = []
    successful = 0
    failed = 0
    succeeded_outputs = []
    failed_tasks = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=True,
        console=Console(force_terminal=True),  # 兼容 rich 14.x
    ) as progress:
        task_progress = progress.add_task(
            "Batch Conversion Progress",
            total=len(tasks)
        )
        for i, task in enumerate(tasks):
            desc = f"Task {i+1}/{len(tasks)}: {task.get('model_path')} → {task.get('output_format')}"
            progress.update(task_progress, description=desc)
            try:
                # 统一输出路径为独立子目录
                task_output_path = auto_complete_output_path(
                    str(task.get('model_path')) if task.get('model_path') is not None else '',
                    str(task.get('output_path')) if task.get('output_path') is not None else None,
                    task.get('output_format')
                )
                result = convert_model(
                    input_path=str(task.get('model_path')) if task.get('model_path') is not None else None,
                    output_path=task_output_path,
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
                    progress.console.print(f"[green]Success! Output: {result.output_path}[/green]")
                    successful += 1
                    succeeded_outputs.append(result.output_path)
                else:
                    progress.console.print(f"[red]Failed: {result.error}[/red]")
                    failed += 1
                    failed_tasks.append(task.get('name') or task.get('model_path') or f'Task {i+1}')
                results.append(result)
            except Exception as e:
                progress.console.print(f"[red]Exception during conversion: {e}[/red]")
                failed += 1
                failed_tasks.append(task.get('name') or task.get('model_path') or f'Task {i+1}')
            progress.advance(task_progress)
    console.print(f"\n[bold]Batch conversion completed: {successful} succeeded, {failed} failed.[/bold]")
    if succeeded_outputs:
        console.print("\n[green]Succeeded output files:[/green]")
        for path in succeeded_outputs:
            console.print(f"  [green]{path}[/green]")
    if failed_tasks:
        console.print("\n[red]Failed tasks:[/red]")
        for name in failed_tasks:
            console.print(f"  [red]{name}[/red]")
    if failed > 0:
        raise typer.Exit(2)

def check_and_handle_disk_space_batch(total_required_bytes: int, console: Console) -> bool:
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
        console.print(f"[green]✓[/green] Sufficient disk space available:")
        console.print(f"  Free: {formatted['free']} | Required: {formatted['required']} | Safety margin: {formatted['safety_margin']}")
        return True
    if space_info["has_enough_for_operation"] and not space_info["has_safety_margin"]:
        return prompt_user_confirmation_low_space(space_info)
    formatted = space_info["formatted"]
    console.print(f"[red]❌[/red] INSUFFICIENT DISK SPACE")
    console.print(f"Current disk space:")
    console.print(f"  Free: {formatted['free']}")
    console.print(f"  Required: {formatted['required']}")
    console.print(f"  Shortage: {format_bytes(total_required_bytes - space_info['free_bytes'])}")
    console.print("Please free up disk space and try again.")
    return False 