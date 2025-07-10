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
    rich_markup_mode="rich"
)

# Subcommand registration
from model_converter_tool.commands import inspect, convert, list_cmd, validate, cache, history, config, version

app.command()(inspect.inspect)
app.command()(convert.convert)
app.command(name="list")(list_cmd.list)
app.command()(validate.validate)
app.command()(cache.cache)
app.command()(history.history)
app.command()(config.config)
app.command()(version.version)

def check_disk_space(min_gb=5):
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024 ** 3)
    if free_gb < min_gb:
        print(f"[ERROR] Not enough disk space. At least {min_gb}GiB required. Aborting.")
        sys.exit(1)

# Call this at the start of main CLI entry
check_disk_space()

if __name__ == "__main__":
    app() 