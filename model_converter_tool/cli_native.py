"""
Model Converter Tool - CLI Native Implementation

This module implements a CLI that is built entirely on top of the API,
following CLI Native principles:
- CLI is a thin wrapper around the API
- Rich CLI-specific features (colors, progress bars, etc.)
- Consistent command structure and help
- Smart defaults and context awareness
"""

import typer
from pathlib import Path
from typing import Optional, List
import rich
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

from .api import ModelConverterAPI, ConversionPlan, ConversionStatus

# Initialize rich console
console = Console()

# Create Typer app
app = typer.Typer(
    name="model-converter",
    help="🚀 Model Converter Tool - Convert models between different formats",
    add_completion=False,
    rich_markup_mode="rich"
)

# Global API instance
api = ModelConverterAPI()


def print_success(message: str):
    """Print success message with rich formatting"""
    console.print(f"✅ {message}", style="green")


def print_error(message: str):
    """Print error message with rich formatting"""
    console.print(f"❌ {message}", style="red")


def print_warning(message: str):
    """Print warning message with rich formatting"""
    console.print(f"⚠️  {message}", style="yellow")


def print_info(message: str):
    """Print info message with rich formatting"""
    console.print(f"ℹ️  {message}", style="blue")


@app.command()
def detect(
    model_path: str = typer.Argument(..., help="Model path or name to detect"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information")
):
    """
    Detect model format and provide detailed information.
    
    Examples:
        model-converter detect meta-llama/Llama-2-7b-hf
        model-converter detect ./my_model --verbose
    """
    with console.status("[bold blue]Detecting model format..."):
        result = api.detect_model(model_path)
    
    if result["detection_confidence"] == "none":
        print_error(f"Failed to detect model format: {result.get('error', 'Unknown error')}")
        raise typer.Exit(1)
    
    # Create rich table for output
    table = Table(title="Model Detection Results")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Format", result["format"])
    table.add_row("Path", result["path"])
    table.add_row("Confidence", result["detection_confidence"])
    
    if verbose and result["supported_outputs"]:
        table.add_row("Supported Outputs", ", ".join(result["supported_outputs"]))
    
    console.print(table)
    
    if result["detection_confidence"] == "low":
        print_warning("Low confidence detection - please verify manually")


@app.command()
def plan(
    model_path: str = typer.Argument(..., help="Input model path or name"),
    output_format: str = typer.Argument(..., help="Target output format"),
    output_path: str = typer.Option(..., "--output", "-o", help="Output file path"),
    model_type: str = typer.Option("auto", "--type", "-t", help="Model type"),
    device: str = typer.Option("auto", "--device", "-d", help="Target device"),
    quantization: Optional[str] = typer.Option(None, "--quantization", "-q", help="Quantization parameters"),
    use_large_calibration: bool = typer.Option(False, "--use-large-calibration", help="Use large calibration dataset for better quantization quality (GPTQ/AWQ only)")
):
    """
    Create a conversion plan without executing it.
    
    This command shows what would be done during conversion,
    including validation results and estimated requirements.
    
    Examples:
        model-converter plan meta-llama/Llama-2-7b-hf gguf ./output.gguf --quantization q4_k_m
        model-converter plan bert-base-uncased onnx ./bert.onnx
    """
    with console.status("[bold blue]Creating conversion plan..."):
        plan = api.plan_conversion(
            model_path=model_path,
            output_format=output_format,
            output_path=output_path,
            model_type=model_type,
            device=device,
            quantization=quantization,
            use_large_calibration=use_large_calibration
        )
    
    # Display plan
    if plan.is_valid:
        console.print(Panel(
            f"[green]✅ Conversion Plan Valid[/green]\n\n"
            f"[bold]Input:[/bold] {plan.model_path}\n"
            f"[bold]Output:[/bold] {plan.output_path}\n"
            f"[bold]Format:[/bold] {plan.output_format}\n"
            f"[bold]Type:[/bold] {plan.model_type}\n"
            f"[bold]Device:[/bold] {plan.device}\n"
            f"[bold]Quantization:[/bold] {plan.quantization or 'None'}\n"
            f"[bold]Large Calibration:[/bold] {'Yes' if getattr(plan, 'use_large_calibration', False) else 'No'}",
            title="Conversion Plan",
            border_style="green"
        ))
        
        if plan.warnings:
            console.print("\n[yellow]⚠️  Warnings:[/yellow]")
            for warning in plan.warnings:
                console.print(f"  • {warning}")
        
        console.print("\n[blue]💡 Run 'model-converter execute' to perform the conversion[/blue]")
    else:
        console.print(Panel(
            f"[red]❌ Conversion Plan Invalid[/red]\n\n"
            f"[bold]Errors:[/bold]\n" + "\n".join(f"  • {error}" for error in plan.errors),
            title="Conversion Plan",
            border_style="red"
        ))
        raise typer.Exit(1)


@app.command()
def execute(
    model_path: str = typer.Argument(..., help="Input model path or name"),
    output_format: str = typer.Argument(..., help="Target output format"),
    output_path: str = typer.Option(..., "--output", "-o", help="Output file path"),
    model_type: str = typer.Option("auto", "--type", "-t", help="Model type"),
    device: str = typer.Option("auto", "--device", "-d", help="Target device"),
    quantization: Optional[str] = typer.Option(None, "--quantization", "-q", help="Quantization parameters"),
    use_large_calibration: bool = typer.Option(False, "--use-large-calibration", help="Use large calibration dataset for better quantization quality (GPTQ/AWQ only)"),
    skip_plan: bool = typer.Option(False, "--skip-plan", help="Skip planning phase")
):
    """
    Execute model conversion.
    
    This command performs the actual model conversion. For large models,
    consider using the plan-execute workflow for safety.
    
    Examples:
        model-converter execute meta-llama/Llama-2-7b-hf gguf ./output.gguf --quantization q4_k_m
        model-converter execute bert-base-uncased onnx ./bert.onnx
    """
    # Create plan first (unless skipped)
    if not skip_plan:
        with console.status("[bold blue]Creating conversion plan..."):
            plan = api.plan_conversion(
                model_path=model_path,
                output_format=output_format,
                output_path=output_path,
                model_type=model_type,
                device=device,
                quantization=quantization,
                use_large_calibration=use_large_calibration
            )
        
        if not plan.is_valid:
            console.print(Panel(
                f"[red]❌ Conversion Plan Invalid[/red]\n\n"
                f"[bold]Errors:[/bold]\n" + "\n".join(f"  • {error}" for error in plan.errors),
                title="Conversion Plan",
                border_style="red"
            ))
            raise typer.Exit(1)
    else:
        # Create plan without validation for skip_plan mode
        plan = ConversionPlan(
            model_path=model_path,
            output_format=output_format,
            output_path=output_path,
            model_type=model_type,
            device=device,
            quantization=quantization,
            use_large_calibration=use_large_calibration
        )
    
    # Execute conversion with progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Converting model...", total=None)
        
        result = api.execute_conversion(plan)
        
        progress.update(task, description="Conversion completed")
    
    # Display results
    if result.success:
        print_success(f"Conversion completed successfully!")
        console.print(f"[bold]Output:[/bold] {result.output_path}")
        
        if result.validation is not None:
            validation_status = "✅ Passed" if result.validation else "❌ Failed"
            console.print(f"[bold]Validation:[/bold] {validation_status}")
    else:
        print_error(f"Conversion failed: {result.error}")
        raise typer.Exit(1)


@app.command()
def convert(
    model_path: str = typer.Argument(..., help="Input model path or name"),
    output_format: str = typer.Argument(..., help="Target output format"),
    output_path: str = typer.Option(..., "--output", "-o", help="Output file path"),
    model_type: str = typer.Option("auto", "--type", "-t", help="Model type"),
    device: str = typer.Option("auto", "--device", "-d", help="Target device"),
    quantization: Optional[str] = typer.Option(None, "--quantization", "-q", help="Quantization parameters"),
    use_large_calibration: bool = typer.Option(False, "--use-large-calibration", help="Use large calibration dataset for better quantization quality (GPTQ/AWQ only)"),
    plan_only: bool = typer.Option(False, "--plan-only", help="Only show conversion plan")
):
    """
    Convert model between formats (convenience command).
    
    This is a convenience command that combines plan and execute.
    For large models, consider using separate plan/execute commands.
    
    Examples:
        model-converter convert meta-llama/Llama-2-7b-hf gguf ./output.gguf --quantization q4_k_m
        model-converter convert bert-base-uncased onnx ./bert.onnx --plan-only
    """
    if plan_only:
        # Just show the plan
        plan(
            model_path=model_path,
            output_format=output_format,
            output_path=output_path,
            model_type=model_type,
            device=device,
            quantization=quantization,
            use_large_calibration=use_large_calibration
        )
    else:
        # Execute directly
        execute(
            model_path=model_path,
            output_format=output_format,
            output_path=output_path,
            model_type=model_type,
            device=device,
            quantization=quantization,
            use_large_calibration=use_large_calibration,
            skip_plan=True
        )


@app.command()
def formats(
    show_matrix: bool = typer.Option(False, "--matrix", "-m", help="Show conversion matrix"),
    input_format: Optional[str] = typer.Option(None, "--input", "-i", help="Show specific input format"),
    output_format: Optional[str] = typer.Option(None, "--output", "-o", help="Show specific output format")
):
    """
    Show supported formats and conversion options.
    
    Examples:
        model-converter formats
        model-converter formats --matrix
        model-converter formats --input huggingface
        model-converter formats --output gguf
    """
    formats_info = api.get_supported_formats()
    
    if input_format:
        # Show specific input format
        input_formats = formats_info["input_formats"]
        if input_format in input_formats:
            console.print(Panel(
                f"[bold]{input_format.upper()}[/bold]\n"
                f"{input_formats[input_format]['description']}\n\n"
                f"[bold]Supported outputs:[/bold]\n" +
                "\n".join(f"  • {fmt}" for fmt in formats_info["conversion_matrix"].get(input_format, [])),
                title=f"Input Format: {input_format}",
                border_style="cyan"
            ))
        else:
            print_error(f"Unknown input format: {input_format}")
            raise typer.Exit(1)
    
    elif output_format:
        # Show specific output format
        output_formats = formats_info["output_formats"]
        if output_format in output_formats:
            fmt_info = output_formats[output_format]
            console.print(Panel(
                f"[bold]{output_format.upper()}[/bold]\n"
                f"{fmt_info['description']}\n\n"
                f"[bold]Quantization:[/bold] {'Supported' if fmt_info['quantization'] else 'Not supported'}",
                title=f"Output Format: {output_format}",
                border_style="green"
            ))
        else:
            print_error(f"Unknown output format: {output_format}")
            raise typer.Exit(1)
    
    elif show_matrix:
        # Show conversion matrix
        matrix = formats_info["conversion_matrix"]
        table = Table(title="Conversion Matrix")
        table.add_column("Input Format", style="cyan")
        table.add_column("Supported Outputs", style="white")
        
        for input_fmt, outputs in matrix.items():
            table.add_row(input_fmt, ", ".join(outputs))
        
        console.print(table)
    
    else:
        # Show overview
        input_formats = formats_info["input_formats"]
        output_formats = formats_info["output_formats"]
        
        console.print(Panel(
            f"[bold]Input Formats:[/bold]\n" +
            "\n".join(f"  • {fmt}: {info['description']}" for fmt, info in input_formats.items()) +
            f"\n\n[bold]Output Formats:[/bold]\n" +
            "\n".join(f"  • {fmt}: {info['description']}" for fmt, info in output_formats.items()),
            title="Supported Formats",
            border_style="blue"
        ))


@app.command()
def status():
    """
    Show current workspace status.
    
    Displays information about the current workspace, including
    any active conversions and available resources.
    """
    workspace_status = api.get_workspace_status()
    
    console.print(Panel(
        f"[bold]Workspace Path:[/bold] {workspace_status.workspace_path}\n\n"
        f"[bold]Active Tasks:[/bold] {len(workspace_status.active_tasks)}\n"
        f"[bold]Completed Tasks:[/bold] {len(workspace_status.completed_tasks)}\n"
        f"[bold]Failed Tasks:[/bold] {len(workspace_status.failed_tasks)}",
        title="Workspace Status",
        border_style="blue"
    ))
    
    if workspace_status.active_tasks:
        console.print("\n[bold]Active Conversions:[/bold]")
        for task in workspace_status.active_tasks:
            console.print(f"  • {task.plan.model_path} → {task.plan.output_format}")


if __name__ == "__main__":
    app() 