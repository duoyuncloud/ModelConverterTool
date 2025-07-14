import typer
import os
from model_converter_tool.core.conversion import convert_model
from model_converter_tool.utils import check_and_handle_disk_space
from model_converter_tool.utils import ansi_safe_wrap
import sys
from rich import print as rprint
import click
import json
import yaml
import shutil

# Beautify parameter help
ARG_REQUIRED = "[bold red][required][/bold red]"
ARG_OPTIONAL = "[dim][optional][/dim]"

def auto_complete_output_path(input_path, output_path, to_format):
    file_exts = {
        'onnx': '.onnx',
        'gguf': '.gguf',
        'pt': '.pt',
        'torchscript': '.pt',
        'safetensors': '.safetensors',
        'fp16': '.safetensors',
        'mlx': '.npz',
    }
    dir_formats = {'hf', 'gptq', 'awq'}
    base = os.path.splitext(os.path.basename(input_path))[0]
    if not output_path:
        if to_format in file_exts:
            return f'./outputs/{base}{file_exts[to_format]}'
        elif to_format in dir_formats:
            return f'./outputs/{base}_{to_format}'
    else:
        if to_format in file_exts:
            ext = file_exts[to_format]
            if not output_path.endswith(ext):
                return output_path + ext
        elif to_format in dir_formats:
            # Remove extension if present
            for ext in file_exts.values():
                if output_path.endswith(ext):
                    return output_path[: -len(ext)]
    return output_path

# Register the command with custom help handling
def convert(
    input: str = typer.Argument(None, help="Input model path or repo id."),
    output: str = typer.Argument(None, help="Output format (e.g. onnx, gguf, custom_quant, etc.)."),
    output_path: str = typer.Option(None, "-o", "--output-path", help="Output file path (auto-completed if omitted)."),
    quant: str = typer.Option(None, help="Quantization type."),
    quant_config: str = typer.Option(None, help="Advanced quantization config (JSON string or YAML file). Supports bits, group_size, sym, desc, etc."),
    model_type: str = typer.Option("auto", help="Model type. Default: auto"),
    device: str = typer.Option("auto", help="Device (cpu/cuda). Default: auto"),
    use_large_calibration: bool = typer.Option(False, help="Use large calibration dataset for quantization. Default: False"),
    dtype: str = typer.Option(None, help="Precision for output weights (e.g., fp16, fp32). Only used for safetensors format."),
):
    """
    Convert models between formats, with optional quantization and precision.

    Examples:
      modelconvert convert bert-base-uncased safetensors --dtype fp16
      modelconvert convert facebook/opt-125m gguf --quant q4_k_m

    Supported Conversion Matrix:
      ┏───────────────┳────────────────────────────────────────────┓
      ┃ Input Format  ┃ Supported Output Formats                   ┃
      ┣───────────────╋────────────────────────────────────────────┫
      ┃ huggingface   ┃ huggingface, safetensors, torchscript,...  ┃
      ┃ safetensors   ┃ huggingface, safetensors                   ┃
      ┃ torchscript   ┃ torchscript                                ┃
      ┃ onnx          ┃ onnx                                       ┃
      ┃ gguf          ┃ gguf                                       ┃
      ┃ mlx           ┃ mlx                                        ┃
      ┗───────────────┻────────────────────────────────────────────┛

    Quantization:
      gptq 4/8bit, awq 4/8bit, gguf q4_k_m/q5_k_m/q8_0, custom_quant any bits/group_size/sym/desc

    Use --quant for simple types, or --quant-config for advanced options.
    """
    # Show concise help if --help, -h, or missing args
    if '--help' in sys.argv or '-h' in sys.argv or not input or not output:
        rprint(convert.__doc__)
        ctx = click.get_current_context()
        typer.echo(ctx.command.get_help(ctx))
        raise typer.Exit()
    # Handle deprecated fp16 format
    if output.lower() == "fp16":
        typer.echo("[yellow]Warning: 'fp16' format is deprecated. Use '--to safetensors --dtype fp16' instead.[/yellow]")
        output = "safetensors"
        if not dtype:
            dtype = "fp16"

    # Check disk space before starting conversion
    if not check_and_handle_disk_space(input, output, quant):
        typer.echo("Conversion aborted due to insufficient disk space.")
        raise typer.Exit(1)

    # Validate model before conversion (integrates validation logic)
    from model_converter_tool.core.validation import validate_model
    val_result = validate_model(input, output)
    if not (isinstance(val_result, dict) and val_result.get('valid', False)):
        errors = val_result.get('errors', [])
        if not errors:
            errors = val_result.get('validation', {}).get('errors', [])
        supported = val_result.get('format_info', {}).get('supported_outputs', [])
        if errors:
            rprint(f"[red]Model validation failed: {'; '.join(errors)}[/red]")
            if supported:
                rprint(f"[yellow]Supported output formats for this model: {', '.join(supported)}[/yellow]")
        else:
            rprint(f"[red]Model validation failed, cannot convert: {val_result}[/red]")
        raise typer.Exit(1)

    output_path = auto_complete_output_path(input, output_path, output)
    quantization_config = None
    if quant_config:
        try:
            if quant_config.strip().endswith(('.yaml', '.yml')) and os.path.exists(quant_config):
                with open(quant_config, 'r') as f:
                    quantization_config = yaml.safe_load(f)
            else:
                quantization_config = json.loads(quant_config)
        except Exception as e:
            typer.echo(f"[red]Failed to parse quantization config: {e}[/red]")
            raise typer.Exit(1)
    result = convert_model(input, output_path, output, quant, model_type, device, use_large_calibration, dtype=dtype, quantization_config=quantization_config)
    typer.echo(f"[Output path used]: {output_path}")
    if result.success:
        typer.echo(f"Conversion succeeded! Output: {result.output_path}")
        if result.validation is not None:
            typer.echo(f"Validation: {'Passed' if result.validation else 'Failed'}")
        if result.extra_info:
            typer.echo(f"Extra info: {result.extra_info}")
    else:
        # Remove invalid output file if it exists (to prevent user confusion)
        import os
        from pathlib import Path
        output_file = Path(output_path)
        if output_file.exists() and output_file.is_file():
            try:
                output_file.unlink()
                typer.echo(f"[Cleanup] Deleted invalid output file: {output_file}")
            except Exception as e:
                typer.echo(f"[Cleanup Warning] Failed to delete invalid output file: {output_file} ({e})")
        typer.echo(f"Conversion failed: {result.error}") 

# Register with Typer, enabling default help
app = typer.Typer()
app.command()(convert) 