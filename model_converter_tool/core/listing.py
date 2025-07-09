from model_converter_tool.api import ModelConverterAPI

def list_supported(target: str = "formats"):
    api = ModelConverterAPI()
    formats = api.get_supported_formats()
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