"""
API example: Use the Model Converter Tool programmatically from Python.
"""
from model_converter_tool.api import ModelConverterAPI

api = ModelConverterAPI()

# Detect model info
info = api.detect_model("path/to/input_model")
print("Model info:", info)

# Convert model to ONNX format
result = api.converter.convert(
    model_name="path/to/input_model",
    output_format="onnx",
    output_path="path/to/output_model.onnx"
)
print("Conversion result:", result) 