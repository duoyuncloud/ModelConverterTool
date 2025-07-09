import typer
from model_converter_tool.core.validation import validate_model

ARG_REQUIRED = "[bold red][required][/bold red]"
ARG_OPTIONAL = "[dim][optional][/dim]"

def validate(
    model: str = typer.Argument(..., help="Model file path."),
    output_format: str = typer.Option(None, help="Target output format.")
):
    """
    [dim]Examples:
      modelconvert validate ./outputs/llama-2-7b.gguf
      modelconvert validate ./outputs/bert.onnx --output-format gguf[/dim]

    Validate a model file or check if conversion is feasible.
    """
    result = validate_model(model, output_format)
    if isinstance(result, dict) and result.get('valid') is not None:
        typer.echo(f"Validation: {'Passed' if result['valid'] else 'Failed'}")
        if 'plan' in result:
            typer.echo(f"Conversion plan: {result['plan']}")
        if 'validation' in result:
            typer.echo(f"Details: {result['validation']}")
    else:
        typer.echo(str(result)) 