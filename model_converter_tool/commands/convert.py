import typer
import os
from model_converter_tool.core.conversion import convert_model

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
    input: str,
    output: str = typer.Option(None, help="Output path (auto-completed if omitted)"),
    to: str = typer.Option(None, help="Output format"),
    quant: str = typer.Option(None, help="Quantization type"),
    model_type: str = typer.Option("auto", help="Model type"),
    device: str = typer.Option("auto", help="Device (cpu/cuda)"),
    use_large_calibration: bool = typer.Option(False, help="Use large calibration dataset for quantization")
):
    """
    Convert a model to another format, with optional quantization.
    Output path will be auto-completed if omitted or mismatched.
    """
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