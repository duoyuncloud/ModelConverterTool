from model_converter_tool.api import ModelConverterAPI

def validate_model(model_path: str, output_format: str = None, **kwargs):
    api = ModelConverterAPI()
    if output_format:
        return api.validate_conversion(model_path, output_format, **kwargs)
    else:
        return api.detect_model(model_path) 