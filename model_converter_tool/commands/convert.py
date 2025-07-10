import typer
import os
from model_converter_tool.core.conversion import convert_model
from model_converter_tool.utils import check_and_handle_disk_space

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


def convert(
    input: str = typer.Argument(..., help="Input model path or repo id."),
    to: str = typer.Option(..., help="Output format."),
    output: str = typer.Option(None, help="Output file path (auto-completed if omitted)."),
    quant: str = typer.Option(None, help="Quantization type."),
    model_type: str = typer.Option("auto", help="Model type. Default: auto"),
    device: str = typer.Option("auto", help="Device (cpu/cuda). Default: auto"),
    use_large_calibration: bool = typer.Option(False, help="Use large calibration dataset for quantization. Default: False")
):
    """
    [dim]Examples:
      modelconvert convert bert-base-uncased --to onnx
      modelconvert convert facebook/opt-125m --to gptq --quant 4bit --output ./outputs/opt_125m_gptq[/dim]

    Output formats: onnx, gguf, torchscript, fp16, gptq, awq, safetensors, mlx

    Convert a model to another format, with optional quantization.
    """
    # Check disk space before starting conversion
    if not check_and_handle_disk_space(input, to, quant):
        typer.echo("Conversion aborted due to insufficient disk space.")
        raise typer.Exit(1)
    
    output_path = auto_complete_output_path(input, output, to)
    result = convert_model(input, output_path, to, quant, model_type, device, use_large_calibration)
    typer.echo(f"[Output path used]: {output_path}")
    if result.success:
        typer.echo(f"Conversion succeeded! Output: {result.output_path}")
        if result.validation is not None:
            typer.echo(f"Validation: {'Passed' if result.validation else 'Failed'}")
        if result.extra_info:
            typer.echo(f"Extra info: {result.extra_info}")
    else:
        typer.echo(f"Conversion failed: {result.error}") 