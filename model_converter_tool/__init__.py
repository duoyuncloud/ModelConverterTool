# 延迟导入，避免在包导入时加载所有引擎模块
__all__ = ["ModelConverter", "ConversionResult", "app"]

def __getattr__(name):
    if name == "ModelConverter":
        from .converter import ModelConverter
        return ModelConverter
    elif name == "ConversionResult":
        from .converter import ConversionResult
        return ConversionResult
    elif name == "app":
        from .cli import app
        return app
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")