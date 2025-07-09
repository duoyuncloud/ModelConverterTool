import typer
from model_converter_tool.core.validation import validate_model

def validate(
    model: str,
    output_format: str = typer.Option(None, help="Target output format (optional)")
):
    """
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