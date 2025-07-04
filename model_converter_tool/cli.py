import os
import sys
import logging
from pathlib import Path
from typing import Any, Optional, Dict, List, Union

import click
import typer
from model_converter_tool.converter import ModelConverter, ConversionResult

# Global ModelConverter instance
converter = ModelConverter()

# Dynamically set help text based on hardware
try:
    import torch

    if torch.cuda.is_available():
        CLI_HELP = "Model Converter CLI (GPU & CPU supported, all formats)"
        CONVERT_HELP = "Convert a model to the specified format (GPU/CPU supported)."
    else:
        CLI_HELP = "Model Converter CLI (CPU supported, all formats)"
        CONVERT_HELP = "Convert a model to the specified format (CPU supported)."
except ImportError:
    CLI_HELP = "Model Converter CLI (CPU supported, all formats)"
    CONVERT_HELP = "Convert a model to the specified format (CPU supported)."


def detect_model_format(input_model):
    """Detect the format of input model."""
    fmt, norm_path = converter._detect_model_format(input_model)
    # norm_path could be path or model name, meta is optional
    meta = {"format": fmt}
    return fmt, norm_path, meta


def load_model_with_fallbacks(norm_path, model_type, device):
    """Load model with fallback strategies."""
    return converter._load_model_with_fallbacks(norm_path, model_type, device)


def validate_conversion_compatibility(in_fmt, output_format, model_type):
    """Validate conversion compatibility."""
    # Simple compatibility validation
    result = converter._validate_conversion_inputs(in_fmt, output_format, model_type, quantization="", device="auto")
    return {
        "compatible": result["valid"],
        "errors": result["errors"],
        "warnings": result["warnings"],
        "recommendations": [],
    }


app = typer.Typer(help="Model Converter Tool CLI (API-First, CLI-Native)")

@app.command()
def convert(
    model_path: str = typer.Argument(..., help="模型名称或路径"),
    output_format: str = typer.Argument(..., help="目标格式 (onnx/gguf/torchscript/fp16/gptq/awq/hf/safetensors/mlx)"),
    output_path: str = typer.Option(..., help="输出文件路径"),
    model_type: str = typer.Option("auto", help="模型类型 (auto/text-generation/...)"),
    device: str = typer.Option("auto", help="设备 (auto/cuda/cpu/mps)"),
    quantization: Optional[str] = typer.Option(None, help="量化参数 (如 q4_k_m)"),
    use_large_calibration: bool = typer.Option(False, help="是否使用高精度大校准集 (量化专用)")
):
    """
    单模型格式转换。所有参数与 API 保持一致。
    """
    converter = ModelConverter()
    # 这里假设用户传入的是模型路径，由 API 内部负责加载
    # 你可以根据实际 API 设计调整为自动加载
    import torch
    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    result: ConversionResult = converter.convert(
        model=model,
        tokenizer=tokenizer,
        model_name=model_path,
            output_format=output_format,
            output_path=output_path,
            model_type=model_type,
            device=device,
        quantization=quantization,
            use_large_calibration=use_large_calibration,
        )
    if result.success:
        typer.echo(f"[SUCCESS] Conversion completed: {result.output_path}")
    else:
        typer.echo(f"[ERROR] Conversion failed: {result.error}")
        raise typer.Exit(code=1)

@app.command()
def batch(
    config_path: str = typer.Argument(..., help="批量任务配置文件 (YAML/JSON)"),
    max_workers: int = typer.Option(1, help="最大并发数"),
    max_retries: int = typer.Option(1, help="最大重试次数"),
):
    """
    批量模型格式转换。配置文件为任务列表。
    """
    import yaml, json
    converter = ModelConverter()
    if config_path.endswith(".yaml") or config_path.endswith(".yml"):
        with open(config_path, "r") as f:
            tasks = yaml.safe_load(f)
    else:
        with open(config_path, "r") as f:
            tasks = json.load(f)
    results = converter.batch_convert(tasks=tasks, max_workers=max_workers, max_retries=max_retries)
    for i, res in enumerate(results):
        if res.success:
            typer.echo(f"[SUCCESS] Task {i+1}: {res.output_path}")
        else:
            typer.echo(f"[ERROR] Task {i+1} failed: {res.error}")

if __name__ == "__main__":
    app()
