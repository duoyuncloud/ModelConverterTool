#!/usr/bin/env python
"""
Model Converter Tool - Unified CLI tool for model format conversion

This tool consolidates model conversion functionality:
- convert: Convert models between different formats
- validate: Validate model files and configurations
- list-formats: Show supported input/output formats
- list-models: Show available model presets
- batch-convert: Convert multiple models from configuration file
"""

import os
import sys
import argparse
from typing import Optional, Dict, Any, List
import logging
import importlib
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from model_converter_tool.converter import ModelConverter
from model_converter_tool.validator import ModelValidator
from model_converter_tool.config import (
    ConversionConfig,
    load_config_preset,
    list_available_presets,
    resolve_final_config,
)
from model_converter_tool.utils import setup_directories, cleanup_temp_files

RECOMMENDED_VERSIONS = {
    "torch": "2.0",
    "transformers": "4.30",
    "onnx": "1.13",
    "onnxruntime": "1.14",
}


def check_dependencies(skip_check=False):
    if skip_check:
        return
    import sys
    import pkg_resources

    print("[Dependency Check] Checking required package versions...")
    for pkg, min_version in RECOMMENDED_VERSIONS.items():
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", None)
            if ver is None:
                ver = pkg_resources.get_distribution(pkg).version
            if tuple(map(int, ver.split(".")[:2])) < tuple(
                map(int, min_version.split("."))
            ):
                print(
                    f"WARNING: {pkg} version {ver} is below recommended {min_version}. Consider upgrading: pip install -U {pkg}"
                )
        except ImportError:
            print(f"WARNING: {pkg} is not installed. Install with: pip install {pkg}")
        except Exception as e:
            print(f"WARNING: Could not check {pkg}: {e}")


def cmd_convert(args) -> None:
    """Convert a model between different formats"""
    try:
        setup_directories()
        # Collect CLI overrides
        cli_overrides = {
            "hidden_size": args.hidden_size,
            "num_hidden_layers": args.num_layers,
            "num_attention_heads": args.num_heads,
            "num_key_value_heads": args.num_kv_heads,
            "vocab_size": args.vocab_size,
            "intermediate_size": args.intermediate_size,
        }
        # Get final config (preset/config-file + CLI overrides)
        config = resolve_final_config(
            getattr(args, "preset", None),
            getattr(args, "config_file", None),
            cli_overrides,
        )
        # Determine input source
        if args.hf_model:
            input_source = f"hf:{args.hf_model}"
            model_name = args.hf_model.split("/")[-1]
        elif args.local_path:
            input_source = args.local_path
            model_name = os.path.basename(args.local_path)
        else:
            print("❌ Either --hf-model or --local-path must be specified")
            sys.exit(1)
        output_path = args.output_path or f"outputs/{model_name}_{args.output_format}"
        print(f"🔄 Converting model: {input_source}")
        print(f"📤 Output format: {args.output_format}")
        print(f"📁 Output path: {output_path}")
        converter = ModelConverter()
        # Pass config as extra argument
        success = converter.convert(
            input_source=input_source,
            output_format=args.output_format,
            output_path=output_path,
            model_type=args.model_type,
            quantization=args.quantization,
            device=args.device,
            config=config,
            offline_mode=getattr(args, "offline_mode", False),
        )
        if success:
            print(f"✅ Conversion completed successfully!")
            print(f"📁 Output saved to: {output_path}")
        else:
            print("❌ Conversion failed")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Conversion error: {e}")
        sys.exit(1)
    finally:
        cleanup_temp_files()


def cmd_validate(args) -> None:
    """Validate model files and configurations"""
    try:
        validator = ModelValidator()

        if args.hf_model:
            model_path = f"hf:{args.hf_model}"
        elif args.local_path:
            model_path = args.local_path
        else:
            print("❌ Either --hf-model or --local-path must be specified")
            sys.exit(1)

        print(f"🔍 Validating model: {model_path}")

        # Run validation
        validation_result = validator.validate_model(
            model_path=model_path,
            model_type=args.model_type,
            check_weights=args.check_weights,
            check_config=args.check_config,
        )

        if validation_result.is_valid:
            print("✅ Model validation passed!")
            if validation_result.details:
                print("📋 Validation details:")
                for detail in validation_result.details:
                    print(f"  - {detail}")
        else:
            print("❌ Model validation failed!")
            if validation_result.errors:
                print("🚨 Validation errors:")
                for error in validation_result.errors:
                    print(f"  - {error}")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Validation error: {e}")
        sys.exit(1)


def cmd_list_formats(args) -> None:
    """List supported input and output formats"""
    try:
        converter = ModelConverter()
        formats = converter.get_supported_formats()

        print("📋 Supported Model Formats:")
        print("=" * 50)

        print("\n🔄 Input Formats:")
        for fmt in formats["input"]:
            print(f"  - {fmt}")

        print("\n📤 Output Formats:")
        for fmt in formats["output"]:
            print(f"  - {fmt}")

        print("\n🔧 Model Types:")
        for model_type in formats["model_types"]:
            print(f"  - {model_type}")

        print("\n⚡ Quantization Options:")
        for quant in formats["quantization"]:
            print(f"  - {quant}")

    except Exception as e:
        print(f"❌ Error listing formats: {e}")
        sys.exit(1)


def cmd_list_models(args) -> None:
    """List available model presets"""
    try:
        list_available_presets()
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        sys.exit(1)


def cmd_batch_convert(args) -> None:
    """Convert multiple models from configuration file, with per-model device/quantization/postprocess and batch validation/logging."""
    if not os.path.exists(args.config_file):
        print(f"❌ Configuration file does not exist: {args.config_file}")
        sys.exit(1)
    try:
        setup_directories()
        converter = ModelConverter()
        validator = ModelValidator()
        # Load batch configuration
        with open(args.config_file, "r") as f:
            import yaml

            batch_config = yaml.safe_load(f)
        print(f"🔄 Starting batch conversion from: {args.config_file}")
        models = batch_config.get("models", {})
        success_count = 0
        total_count = len(models)
        os.makedirs("outputs/logs", exist_ok=True)
        for model_name, model_config in models.items():
            print(f"\n📦 Processing: {model_name}")
            log_lines = []
            log_lines.append(f"[{datetime.datetime.now()}] Processing: {model_name}")
            try:
                # Extract conversion parameters
                input_source = model_config.get("input")
                output_format = model_config.get("output_format")
                output_path = model_config.get("output_path")
                model_type = model_config.get("model_type", "auto")
                quantization = model_config.get("quantization")
                device = model_config.get("device", "auto")
                postprocess = model_config.get("postprocess")
                offline_mode = getattr(args, "offline_mode", False)
                # Perform conversion
                log_lines.append(
                    f"Converting: {input_source} -> {output_format} at {output_path}"
                )
                result = converter.convert(
                    input_source=input_source,
                    output_format=output_format,
                    output_path=output_path,
                    model_type=model_type,
                    quantization=quantization,
                    device=device,
                    offline_mode=offline_mode,
                    postprocess=postprocess,
                )
                if result.get("success"):
                    print(f"✅ {model_name}: Conversion successful")
                    log_lines.append("Conversion successful")
                    # Validate output
                    val_result = validator.validate_model(
                        model_path=output_path, model_type=model_type
                    )
                    if val_result.is_valid:
                        print(f"  - Validation passed!")
                        log_lines.append("Validation passed!")
                    else:
                        print(f"  - Validation failed!")
                        log_lines.append("Validation failed!")
                        for err in val_result.errors:
                            print(f"    - {err}")
                            log_lines.append(f"    - {err}")
                    # Print details/warnings
                    for detail in val_result.details:
                        log_lines.append(f"    {detail}")
                    for warn in val_result.warnings:
                        log_lines.append(f"    WARNING: {warn}")
                    success_count += 1 if val_result.is_valid else 0
                else:
                    print(f"❌ {model_name}: Conversion failed")
                    log_lines.append("Conversion failed")
                # Postprocess summary
                if postprocess:
                    print(
                        f"  - Postprocess result: {result.get('postprocess_result') or '-'}"
                    )
                    log_lines.append(
                        f"Postprocess result: {result.get('postprocess_result') or '-'}"
                    )
            except Exception as e:
                print(f"❌ {model_name}: Error - {e}")
                log_lines.append(f"Error: {e}")
            # Write log for this model
            log_path = f"outputs/logs/{model_name}.log"
            with open(log_path, "w") as lf:
                lf.write("\n".join(log_lines))
        print(
            f"\n📊 Batch conversion completed: {success_count}/{total_count} successful"
        )
        if success_count < total_count:
            sys.exit(1)
    except Exception as e:
        print(f"❌ Batch conversion error: {e}")
        sys.exit(1)
    finally:
        cleanup_temp_files()


def main():
    """Main CLI entry point"""
    skip_dep_check = "--skip-dependency-check" in sys.argv
    check_dependencies(skip_check=skip_dep_check)
    parser = argparse.ArgumentParser(
        description="Model Converter Tool - Convert models between different formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert HuggingFace model to ONNX using a preset
  python model_converter.py convert --hf-model gpt2 --output-format onnx --preset my_preset
  # Convert with custom config file
  python model_converter.py convert --hf-model gpt2 --output-format onnx --config-file my_config.yaml
  # Override architecture params
  python model_converter.py convert --hf-model gpt2 --output-format onnx --hidden-size 2048 --num-layers 24
  # Offline mode (local only)
  python model_converter.py convert --local-path /models/my_model --output-format onnx --offline-mode
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.required = True

    # convert command
    parser_convert = subparsers.add_parser(
        "convert", help="Convert model between formats"
    )

    # Input source (mutually exclusive)
    input_group = parser_convert.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--hf-model",
        type=str,
        help="HuggingFace model name (e.g., gpt2, bert-base-uncased)",
    )
    input_group.add_argument(
        "--local-path", type=str, help="Path to local model directory or file"
    )

    parser_convert.add_argument(
        "--output-format",
        type=str,
        required=True,
        help="Target output format (e.g., onnx, gguf, mlx)",
    )
    parser_convert.add_argument(
        "--output-path",
        type=str,
        help="Output path (default: outputs/<model_name>_<format>)",
    )
    parser_convert.add_argument(
        "--model-type",
        type=str,
        default="auto",
        help="Model type (auto, text-generation, text-classification, etc.)",
    )
    parser_convert.add_argument(
        "--quantization", type=str, help="Quantization method (q4_k_m, q8_0, etc.)"
    )
    parser_convert.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for conversion (auto, cpu, cuda)",
    )
    # Preset/config-file/architecture overrides
    parser_convert.add_argument(
        "--preset", type=str, help="Model configuration preset name"
    )
    parser_convert.add_argument(
        "--config-file", type=str, help="Path to YAML configuration file"
    )
    parser_convert.add_argument("--hidden-size", type=int, help="Override hidden size")
    parser_convert.add_argument(
        "--num-layers", type=int, help="Override number of layers"
    )
    parser_convert.add_argument(
        "--num-heads", type=int, help="Override number of attention heads"
    )
    parser_convert.add_argument(
        "--num-kv-heads", type=int, help="Override number of key-value heads"
    )
    parser_convert.add_argument(
        "--vocab-size", type=int, help="Override vocabulary size"
    )
    parser_convert.add_argument(
        "--intermediate-size", type=int, help="Override intermediate size"
    )
    parser_convert.add_argument(
        "--offline-mode",
        action="store_true",
        help="Enable offline mode (only use local files, no downloads)",
    )
    parser_convert.set_defaults(func=cmd_convert)

    # validate command
    parser_validate = subparsers.add_parser(
        "validate", help="Validate model files and configurations"
    )

    # Input source (mutually exclusive)
    validate_input_group = parser_validate.add_mutually_exclusive_group(required=True)
    validate_input_group.add_argument(
        "--hf-model", type=str, help="HuggingFace model name"
    )
    validate_input_group.add_argument(
        "--local-path", type=str, help="Path to local model directory or file"
    )

    parser_validate.add_argument(
        "--model-type", type=str, default="auto", help="Model type for validation"
    )
    parser_validate.add_argument(
        "--check-weights", action="store_true", help="Check model weights integrity"
    )
    parser_validate.add_argument(
        "--check-config", action="store_true", help="Check model configuration"
    )
    parser_validate.set_defaults(func=cmd_validate)

    # list-formats command
    parser_formats = subparsers.add_parser(
        "list-formats", help="List supported formats"
    )
    parser_formats.set_defaults(func=cmd_list_formats)

    # list-models command
    parser_models = subparsers.add_parser(
        "list-models", help="List available model presets"
    )
    parser_models.set_defaults(func=cmd_list_models)

    # batch-convert command
    parser_batch = subparsers.add_parser(
        "batch-convert", help="Convert multiple models from config file"
    )
    parser_batch.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Path to batch configuration YAML file",
    )
    parser_batch.add_argument(
        "--offline-mode",
        action="store_true",
        help="Enable offline mode for all batch conversions (only use local files, no downloads)",
    )
    parser_batch.set_defaults(func=cmd_batch_convert)

    # Parse arguments and execute command
    args = parser.parse_args()

    # Ensure we're in the correct directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
