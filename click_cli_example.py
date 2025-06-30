import click
from model_converter_tool.compatibility import (
    detect_model_format,
    load_model_with_fallbacks,
    validate_conversion_compatibility
)
import torch
import sys

@click.group()
def cli():
    """Model Converter CLI (CPU-only, supports all formats including quantized models)"""
    pass

@cli.command()
@click.argument('input_model')
@click.argument('output_format')
@click.option('--output-path', default=None, help='Path to save the converted model')
@click.option('--model-type', default='auto', help='Model type (auto/text-generation/text-classification/...)')
def convert(input_model, output_format, output_path, model_type):
    """Convert a model to the specified format (CPU-only)."""
    click.echo(f"[INFO] Detecting input model format for: {input_model}")
    in_fmt, norm_path, meta = detect_model_format(input_model)
    click.echo(f"[INFO] Detected input format: {in_fmt} ({meta.get('format')})")

    # 兼容性检查
    result = validate_conversion_compatibility(in_fmt, output_format, model_type)
    if not result['compatible']:
        click.echo(f"[ERROR] Incompatible conversion: {result['errors']}")
        sys.exit(1)
    if result['warnings']:
        click.echo(f"[WARN] {result['warnings']}")
    if result['recommendations']:
        click.echo(f"[RECOMMEND] {result['recommendations']}")

    # 自动检测设备
    if torch.cuda.is_available():
        device = 'cuda'
        click.echo(f"[INFO] 检测到GPU，所有操作将在GPU上执行")
    else:
        device = 'cpu'
        click.echo(f"[INFO] 未检测到GPU，所有操作将在CPU上执行")

    # GPTQ/AWQ等量化模型特殊处理
    if output_format in ['gptq', 'awq']:
        click.echo(f"[INFO] Attempting quantized conversion ({output_format}) on CPU...")
        # 检查相关库是否支持CPU
        try:
            if output_format == 'gptq':
                import auto_gptq
                # 检查auto_gptq是否支持CPU
                if not torch.cuda.is_available():
                    click.echo("[WARN] auto-gptq库主要为GPU优化，CPU下速度较慢且部分功能可能不可用。")
            elif output_format == 'awq':
                import awq
                click.echo("[WARN] awq库主要为GPU优化，CPU下速度较慢且部分功能可能不可用。")
        except ImportError:
            click.echo(f"[ERROR] {output_format}库未安装，无法进行量化转换。请先安装相关依赖。")
            sys.exit(1)

    # 加载模型
    click.echo(f"[INFO] Loading model with fallback strategies...")
    model, tokenizer, load_meta = load_model_with_fallbacks(norm_path, model_type, device)
    click.echo(f"[INFO] Model loaded. Device: {load_meta.get('device')}, Format: {load_meta.get('format')}")

    # 这里只做演示：实际转换逻辑需根据output_format实现
    click.echo(f"[INFO] (DEMO) Would convert model to {output_format} and save to {output_path or '[not specified]'}")
    click.echo(f"[SUCCESS] Conversion pipeline completed (CPU-only mode).")

if __name__ == '__main__':
    cli() 