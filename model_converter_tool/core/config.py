from model_converter_tool.config import ConfigManager

def manage_config(action: str = "show", key: str = None, value: str = None):
    mgr = ConfigManager()
    if action == "show":
        return mgr.config
    elif action == "get" and key:
        return mgr.get(key)
    elif action == "set" and key:
        mgr.set(key, value)
        return mgr.get(key)
    elif action == "list_presets":
        return mgr.list_presets()
    else:
        return {"error": "Unknown config action"} 