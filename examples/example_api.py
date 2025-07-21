"""
API example: Use the Model Converter Tool programmatically from Python.
"""

from model_converter_tool.api import ModelConverterAPI

api = ModelConverterAPI()

# Detect model info
info = api.detect_model("gpt2")
print("Model info:", info)

# Convert model to ONNX format
result = api.convert_model(model_path="gpt2", output_format="onnx", output_path="outputs/gpt2.onnx")
print("ONNX conversion result:", result)

# Convert a muP model to LLaMA format with scaling (if muP params present)
# result = api.convert_model(
#     model_path="path/to/mup_model",
#     output_format="safetensors",
#     output_path="outputs/mup2llama",
#     mup2llama=True
# )
# print("muP-to-LLaMA result:", result)

# Convert a model with fake weights (for testing)
# result = api.convert_model(
#     model_path="gpt2",
#     output_format="safetensors",
#     output_path="outputs/gpt2_fake",
#     fake_weight=True
# )
# print("Fake weight result:", result)
