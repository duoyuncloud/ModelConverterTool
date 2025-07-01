from setuptools import setup, find_packages
import os

# Try to read README.md, fallback to description if not found
try:
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "A CLI and API tool for converting machine learning models between formats."

setup(
    name="model-converter-tool",
    version="1.0.0",
    description="A CLI and API tool for converting machine learning models between formats.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/duoyuncloud/ModelConverterTool",
    packages=find_packages(include=["model_converter_tool", "model_converter_tool.*"]),
    install_requires=[
        "torch",
        "transformers",
        "onnx",
        "onnxruntime",
        "safetensors",
        # 其他依赖...
    ],
    python_requires=">=3.8",
    include_package_data=True,
    entry_points={"console_scripts": ["model-converter = click_cli_example:cli"]},
)
