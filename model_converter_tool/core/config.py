from model_converter_tool.api import ModelConverterAPI

def manage_config(action: str = "show", key: str = None, value: str = None):
    """
    Manage configuration using the API layer. Supports actions: show, get, set, list_presets.
    """
    api = ModelConverterAPI()
    return api.manage_config(action, key, value) 