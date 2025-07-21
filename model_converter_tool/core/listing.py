from model_converter_tool.api import ModelConverterAPI


def list_supported(target: str = "formats"):
    """
    List supported formats or quantizations using the API layer.
    """
    api = ModelConverterAPI()
    formats = api.list_supported()
    if target == "formats":
        return formats
    elif target == "quantizations":
        quant = {}
        for fmt, info in formats["output_formats"].items():
            if info.get("quantization"):
                quant[fmt] = info.get("quantization_options", [])
        return quant
    else:
        return formats
