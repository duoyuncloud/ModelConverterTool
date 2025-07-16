from model_converter_tool.api import ModelConverterAPI

def validate_model(model_path: str, output_format: str = None, **kwargs):
    """
    Static validation: check if input/output formats and parameters are valid and convertible.
    If output_format is not provided, only detect the model format.
    """
    api = ModelConverterAPI()
    if output_format:
        return api.validate_conversion(model_path, output_format, **kwargs)
    else:
        return api.detect_model(model_path) 