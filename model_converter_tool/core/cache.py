from model_converter_tool.api import ModelConverterAPI

def manage_cache(action: str = "show"):
    """
    Manage cache using the API layer. Only 'show' is implemented.
    """
    api = ModelConverterAPI()
    if action == "show":
        return api.manage_cache(action)
    return {"action": action, "result": "not implemented"} 