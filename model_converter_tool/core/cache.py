from model_converter_tool.api import ModelConverterAPI

def manage_cache(action: str = "show"):
    api = ModelConverterAPI()
    if action == "show":
        return api.get_workspace_status()
    return {"action": action, "result": "not implemented"} 