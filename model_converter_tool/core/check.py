from model_converter_tool.api import ModelConverterAPI


def check_model(model_path: str, model_format: str = None, **kwargs):
    """
    Dynamic check: actually load the model and try inference, return if inference is possible.
    """
    api = ModelConverterAPI()
    result = api.check_model(model_path, model_format, **kwargs)
    if result.get("error"):
        print(f"[CHECK DEBUG] Error: {result['error']}")
    if result.get("details"):
        print(f"[CHECK DEBUG] Details: {result['details']}")
    return result
