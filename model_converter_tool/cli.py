import typer
from rich import print as rprint
import os
import shutil
import sys
# os.environ["HF_ENDPOINT"] = "https://huggingface.co"

EXAMPLES = """
[bold cyan]Examples:[/bold cyan]
  modelconvert inspect meta-llama/Llama-2-7b-hf
  modelconvert convert bert-base-uncased --to onnx
  modelconvert convert facebook/opt-125m --to gptq --quant 4bit
  modelconvert batch configs/batch_template.yaml
  modelconvert list formats
  modelconvert validate ./outputs/llama-2-7b.gguf
  modelconvert cache
  modelconvert config set cache_dir ./mycache
"""

# CLI app
app = typer.Typer(
    help=f"""
{EXAMPLES}
[bold]Model Converter Tool[/bold] - Professional, Clean CLI

A professional, multi-format machine learning model conversion and management tool.
Supports ONNX, GGUF, MLX, TorchScript, FP16, GPTQ, AWQ, SafeTensors, HuggingFace and more.
Built-in quantization, validation, and batch processing.

Run [green]modelconvert --help[/green] or [green]modelconvert <command> --help[/green] for details.
""",
    rich_markup_mode="rich",
    add_completion=False,
)

# Add --version support
app = typer.Typer(
    help=app.help,
    rich_markup_mode="rich",
    add_completion=False,
    no_args_is_help=True,
)

def version_callback(value: bool):
    if value:
        typer.echo("modelconvertertool version 1.0.0")
        raise typer.Exit()

app.callback()(typer.Option(None, "--version", callback=version_callback, is_eager=True, help="Show the tool version and exit."))

# Subcommand registration
from model_converter_tool.commands import inspect, convert, list_cmd, history, config, batch

app.command()(inspect.inspect)
app.command()(convert.convert)
app.command()(batch.batch)
app.command()(history.history)
app.command()(config.config)

if __name__ == "__main__":
    app() 