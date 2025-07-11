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
    output: str = typer.Argument(..., help="Output format."),
    path: str = typer.Option(None, help="Output file path (auto-completed if omitted)."),
    quant: str = typer.Option(None, help="Quantization type."),
    model_type: str = typer.Option("auto", help="Model type. Default: auto"),
    device: str = typer.Option("auto", help="Device (cpu/cuda). Default: auto"),
    use_large_calibration: bool = typer.Option(False, help="Use large calibration dataset for quantization. Default: False")
):
    """
    [dim]Examples:
      modelconvert convert bert-base-uncased onnx
      modelconvert convert facebook/opt-125m gptq --quant 4bit --path ./outputs/opt_125m_gptq[/dim]

    Output formats: onnx, gguf, torchscript, fp16, gptq, awq, safetensors, mlx

    Supported conversion matrix:

      Input Format   | Supported Output Formats
      --------------|----------------------------------------------------------
      huggingface   | onnx, gguf, torchscript, fp16, gptq, awq, safetensors, mlx, hf
      safetensors   | onnx, gguf, torchscript, fp16, gptq, awq, safetensors, mlx, hf
      torchscript   | onnx, torchscript, hf
      onnx          | onnx, hf
      gguf          | gguf, hf
      mlx           | mlx, hf

    Supported quantization types:
      - gptq: 4bit, 8bit
      - awq: 4bit, 8bit
      - gguf: q4_k_m, q4_k_s, q5_k_m, q5_k_s, q6_k, q8_0
      - mlx: q4_k_m, q8_0, q5_k_m

    Convert a model to another format, with optional quantization.
    """
    # Check disk space before starting conversion
    if not check_and_handle_disk_space(input, output, quant):
        typer.echo("Conversion aborted due to insufficient disk space.")
        raise typer.Exit(1)

    # 在转换前做模型有效性和可转化性检查（集成validate逻辑）
    from model_converter_tool.core.validation import validate_model
    val_result = validate_model(input, output)
    if not (isinstance(val_result, dict) and val_result.get('valid', False)):
        errors = val_result.get('errors', [])
        if errors:
            typer.echo(f"[red]Model validation failed, cannot convert: {'; '.join(errors)}[/red]")
        else:
            typer.echo(f"[red]Model validation failed, cannot convert: {val_result}[/red]")
        raise typer.Exit(1)

    output_path = auto_complete_output_path(input, path, output)
    result = convert_model(input, output_path, output, quant, model_type, device, use_large_calibration)
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