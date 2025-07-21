import typer
import os
from model_converter_tool.core.convert import convert_model
from model_converter_tool.utils import check_and_handle_disk_space
from model_converter_tool.utils import ansi_safe_wrap
import sys
from rich import print as rprint
import click
import json
import yaml
import shutil
from model_converter_tool.api import ModelConverterAPI
from pathlib import Path
import re
from model_converter_tool.utils import auto_complete_output_path

# Dynamically generate a beautified conversion matrix table (pure text, English comments)
def get_conversion_matrix_table():
    api = ModelConverterAPI()
    matrix = api.list_supported(full=True)["conversion_matrix"]
    # Calculate the max width for input format column
    max_input_len = max(len(fmt) for fmt in matrix)
    header = f"{'Input Format'.ljust(max_input_len)} | Supported Output Formats"
    separator = f"{'-' * max_input_len}-+-{'-' * 28}"
    lines = [header, separator]
    for input_fmt, outputs in matrix.items():
        out_str = ", ".join(outputs)
        lines.append(f"{input_fmt.ljust(max_input_len)} | {out_str}")
    return '\n'.join(lines)

CONVERSION_MATRIX = get_conversion_matrix_table()

EXAMPLES = f"""
[bold cyan]Examples:[/bold cyan]
  modelconvert convert bert-base-uncased --to onnx
  modelconvert convert facebook/opt-125m --to gptq --quant 4bit
  modelconvert convert sshleifer/tiny-gpt2 --to safetensors --dtype fp16 -o ./outputs/tiny_gpt2_fp16

Supported Conversion Matrix:
{CONVERSION_MATRIX}
"""

ARG_REQUIRED = "[bold red][required][/bold red]"
ARG_OPTIONAL = "[dim][optional][/dim]"


def convert(
    input: str = typer.Argument(None, help="Input model path or repo id."),
    output: str = typer.Argument(None, help="Output format (e.g. onnx, gguf, etc.)."),
    output_path: str = typer.Option(None, "-o", "--output-path", help="Output file path (auto-completed if omitted)."),
    quant: str = typer.Option(None, help="Quantization type."),
    quant_config: str = typer.Option(
        None,
        help="""Advanced quantization config (JSON string or YAML file). See README for details."""
    ),
    model_type: str = typer.Option("auto", help="Model type. Default: auto"),
    device: str = typer.Option("auto", help="Device (cpu/cuda). Default: auto"),
    use_large_calibration: bool = typer.Option(False, help="Use large calibration dataset for quantization. Default: False"),
    dtype: str = typer.Option(None, help="Precision for output weights (e.g., fp16, fp32). Only used for safetensors format."),
    fake_weight: bool = typer.Option(False, help="Use fake weights for the model (for testing and debugging). Default: False"),
    fake_weight_config: str = typer.Option(
        None,
        help="Path to a JSON or YAML file specifying custom shapes for fake weights. Example: embed_tokens.weight: [32000, 4096] (YAML) or {\"embed_tokens.weight\": [32000, 4096]} (JSON). Overrides default shapes if provided."
    ),
    mup2llama: bool = typer.Option(False, help="Enable muP-to-LLaMA parameter scaling during conversion."),
):
    """
    Convert models between formats, with optional quantization, precision, and fake weights.

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
    # Alias mapping for output format
    output_aliases = {"hf": "huggingface"}
    output = output_aliases.get(output.lower(), output.lower())
    if '--help' in sys.argv or '-h' in sys.argv or not input or not output:
        rprint(convert.__doc__)
        ctx = click.get_current_context()
        typer.echo(ctx.command.get_help(ctx))
        raise typer.Exit()
    if output.lower() == "fp16":
        typer.echo("[yellow]Warning: 'fp16' format is deprecated. Use '--to safetensors --dtype fp16' instead.[/yellow]")
        output = "safetensors"
        if not dtype:
            dtype = "fp16"

    if not check_and_handle_disk_space(input, output, quant):
        typer.echo("Conversion aborted due to insufficient disk space.")
        raise typer.Exit(1)

    # Remove the call to validate_model (pre-validation) and related error handling.
    # Only call convert_model (which already includes validation) and handle its result.
    # Keep error messages and output user-friendly.
    output_path = auto_complete_output_path(input, output_path, output)
    # 明确提示实际输出目录
    typer.echo(f"[Output directory used]: {output_path}")
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

    # Parse fake_weight_config if provided
    fake_weight_shape_dict = None  # This will hold the parsed config
    if fake_weight_config:
        try:
            # Support both YAML and JSON
            if fake_weight_config.strip().endswith((".yaml", ".yml")) and os.path.exists(fake_weight_config):
                with open(fake_weight_config, 'r') as f:
                    fake_weight_shape_dict = yaml.safe_load(f)
            elif fake_weight_config.strip().endswith(".json") and os.path.exists(fake_weight_config):
                with open(fake_weight_config, 'r') as f:
                    fake_weight_shape_dict = json.load(f)
            else:
                raise ValueError("Unsupported fake_weight_config file type. Please use .json or .yaml/.yml.")
            # --- Enhanced validation ---
            if not isinstance(fake_weight_shape_dict, dict):
                raise ValueError("fake_weight_config must be a dict mapping parameter names to shape lists.")
            for k, v in fake_weight_shape_dict.items():
                if not (isinstance(v, (list, tuple)) and all(isinstance(i, int) and i > 0 for i in v)):
                    raise ValueError(
                        f"Invalid shape for '{k}': {v}. Each shape must be a list of positive integers, e.g. [4096, 4096]."
                    )
        except Exception as e:
            typer.echo(f"[red]Failed to parse fake_weight_config: {e}[/red]")
            typer.echo("[yellow]Example YAML:\nembed_tokens.weight: [32000, 4096]\nlayers.0.self_attn.q_proj.weight: [4096, 4096]\n\nExample JSON:\n{\"embed_tokens.weight\": [32000, 4096]}[/yellow]")
            raise typer.Exit(1)

    result = convert_model(
        input,
        output_path,
        output,
        quant,
        model_type,
        device,
        use_large_calibration,
        dtype=dtype,
        quantization_config=quantization_config,
        fake_weight=fake_weight,
        fake_weight_shape_dict=fake_weight_shape_dict,  # Pass the parsed config to the core logic
        mup2llama=mup2llama,
    )
    if result.success:
        typer.echo(f"Conversion succeeded! Output: {result.output_path}")
        if result.validation is not None:
            typer.echo(f"Validation: {'Passed' if result.validation else 'Failed'}")
        if result.extra_info:
            typer.echo(f"Extra info: {result.extra_info}")
    else:
        # Only attempt cleanup if output_path is not None
        if result.output_path:
            from pathlib import Path
            output_file = Path(result.output_path)
            if output_file.exists() and output_file.is_file():
                try:
                    output_file.unlink()
                    typer.echo(f"[Cleanup] Deleted invalid output file: {output_file}")
                except Exception as e:
                    typer.echo(f"[Cleanup Warning] Failed to delete invalid output file: {output_file} ({e})")
        typer.echo(f"Conversion failed: {result.error}")
        # Ensure CLI returns non-zero exit code on failure for CI/scripting compatibility
        raise typer.Exit(1) 