from model_converter_tool.api import ModelConverterAPI


def inspect_model(model_path: str):
    api = ModelConverterAPI()
    return api.detect_model(model_path)
