import typer
from model_converter_tool.engine.gguf import convert_to_gguf
from model_converter_tool.utils import auto_load_model_and_tokenizer
from rich import print as rprint
import sys
import json
import yaml
import os


def to_llama_format(
    input: str = typer.Argument(..., help="Input model path or repo id."),
    output_path: str = typer.Option(None, "-o", "--output-path", help="Output file path (auto-completed if omitted)."),
    quant: str = typer.Option(None, help="Quantization type (e.g. q4_k_m, q8_0, f16, auto)."),
    quant_config: str = typer.Option(None, help="Advanced quantization config (JSON string or YAML file)."),
    model_type: str = typer.Option("auto", help="Model type. Default: auto"),
    device: str = typer.Option("auto", help="Device (cpu/cuda). Default: auto"),
):
    """
    Convert a model to llama.cpp GGUF format (to-llama-format).

    Example:
      modelconvert to-llama-format meta-llama/Llama-2-7b-hf -o ./outputs/llama-2-7b.gguf --quant q4_k_m
    """
    if '--help' in sys.argv or '-h' in sys.argv or not input:
        rprint(to_llama_format.__doc__)
        raise typer.Exit()

    # Auto-complete output path
    def auto_complete_output_path(input_path, output_path):
        base = os.path.splitext(os.path.basename(input_path))[0]
        if not output_path:
            return f'./outputs/{base}.gguf'
        if not output_path.endswith('.gguf'):
            return output_path + '.gguf'
        return output_path

    output_path = auto_complete_output_path(input, output_path)

    quantization_config = None
    if quant_config:
        try:
            if quant_config.strip().endswith(('.yaml', '.yml')) and os.path.exists(quant_config):
                with open(quant_config, 'r') as f:
                    quantization_config = yaml.safe_load(f)
            else:
                quantization_config = json.loads(quant_config)
        except Exception as e:
            rprint(f"[red]Failed to parse quantization config: {e}[/red]")
            raise typer.Exit(1)

    # Load model and tokenizer
    model, tokenizer = auto_load_model_and_tokenizer(None, None, input, model_type)
    success, extra = convert_to_gguf(
        model, tokenizer, input, output_path, model_type, device, quant, False, quantization_config
    )
    if success:
        rprint(f"[green]Conversion succeeded! Output: {output_path}[/green]")
    else:
        rprint(f"[red]Conversion failed.[/red]")
        if extra:
            rprint(f"[red]{extra}[/red]") 