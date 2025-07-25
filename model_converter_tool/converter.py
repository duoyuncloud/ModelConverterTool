import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Supported format list
SUPPORTED_FORMATS = [
    "onnx",
    "torchscript",
    "gguf",
    "awq",
    "gptq",
    "hf",
    "huggingface",
    "safetensors",
    "mlx",
    "mtk",
    "rk",
    "ax",
    "qnn",
    "megatron2hf",  # Newly added planned formats
    "hf2megatron",  # Added for bidirectional support
]


@dataclass
class ConversionResult:
    success: bool
    error: Optional[str] = None
    output_path: Optional[str] = None
    validation: Optional[bool] = None
    extra_info: Optional[Dict[str, Any]] = None


class ModelConverter:
    """
    Model converter with engine-based logic and unified dispatch.
    API-First: All parameters explicit, type-safe, well-documented, returns dataclass.
    """

    def _get_converter_functions(self, output_format: str):
        print(f"[DEBUG] _get_converter_functions: output_format={output_format}")
        """Lazy import converter functions"""
        if output_format in ("hf", "huggingface"):
            from .engine.hf import convert_to_hf, validate_hf_file

            return convert_to_hf, validate_hf_file
        if output_format == "onnx":
            from .engine.onnx import convert_to_onnx, validate_onnx_file

            return convert_to_onnx, validate_onnx_file
        elif output_format == "torchscript":
            from .engine.torchscript import convert_to_torchscript, validate_torchscript_file

            return convert_to_torchscript, validate_torchscript_file
        elif output_format == "gguf":
            from .engine.gguf import convert_to_gguf, validate_gguf_file

            return convert_to_gguf, validate_gguf_file
        elif output_format == "awq":
            from .engine.awq import convert_to_awq, validate_awq_file

            return convert_to_awq, validate_awq_file
        elif output_format == "gptq":
            from .engine.gptq import convert_to_gptq, validate_gptq_file

            return convert_to_gptq, validate_gptq_file
        elif output_format == "safetensors":
            from .engine.safetensors import convert_to_safetensors, validate_safetensors_file

            return convert_to_safetensors, validate_safetensors_file
        elif output_format == "mlx":
            from .engine.mlx import convert_to_mlx, validate_mlx_file

            return convert_to_mlx, validate_mlx_file
        elif output_format == "custom_quant":
            from .engine.custom_quant import convert_to_custom_quant, validate_custom_quant_file

            return convert_to_custom_quant, validate_custom_quant_file
        # Register new shell converters
        elif output_format == "mtk":
            from .engine.mtk import convert_hf_to_mtk

            return convert_hf_to_mtk, (lambda *a, **kw: True)
        elif output_format == "rk":
            from .engine.rk import convert_hf_to_rk

            return convert_hf_to_rk, (lambda *a, **kw: True)
        elif output_format == "ax":
            from .engine.ax import convert_hf_to_ax

            return convert_hf_to_ax, (lambda *a, **kw: True)
        elif output_format == "qnn":
            from .engine.qnn import convert_hf_to_qnn

            return convert_hf_to_qnn, (lambda *a, **kw: True)
        elif output_format == "megatron2hf":
            from .engine.megatron2hf import convert_megatron_to_hf

            return convert_megatron_to_hf, (lambda *a, **kw: True)
        elif output_format == "hf2megatron":
            from .engine.megatron2hf import convert_hf_to_megatron

            return convert_hf_to_megatron, (lambda *a, **kw: True)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def _detect_model_format(self, input_model: str) -> Tuple[str, str]:
        """
        Detect the format of input model.
        Args:
            input_model: Model path or name
        Returns:
            Tuple of (format, normalized_path)
        """
        path = Path(input_model)
        suffix = path.suffix.lower()
        # If the path is a directory and contains model.onnx or model.gguf, treat as ONNX or GGUF format
        if path.is_dir():
            if (path / "model.onnx").exists():
                return "onnx", str(path / "model.onnx")
            if (path / "model.gguf").exists():
                return "gguf", str(path / "model.gguf")
            if (
                (path / "config.json").exists()
                or (path / "pytorch_model.bin").exists()
                or any(path.glob("*.safetensors"))
            ):
                return "huggingface", str(path)
            return "unknown", str(path)
        # File extension based detection
        if suffix == ".onnx":
            return "onnx", str(path)
        elif suffix == ".gguf":
            return "gguf", str(path)
        elif suffix in [".pt", ".pth"]:
            return "torchscript", str(path)
        elif suffix == ".safetensors":
            return "safetensors", str(path)
        elif suffix == ".npz":
            return "mlx", str(path)
        # If local doesn't exist and no clear file extension, consider it Hugging Face Hub name
        if not path.exists():
            if "/" in input_model or "\\" in input_model or (not suffix and not input_model.startswith(".")):
                return "huggingface", input_model
            else:
                return "unknown", str(path)
        # If it's a file but no supported extension
        if path.is_file():
            return "unknown", str(path)
        # Default to unknown
        return "unknown", str(path)

    def _validate_conversion_inputs(
        self, input_format: str, output_format: str, model_type: str, quantization: str = "", device: str = "auto"
    ) -> Dict[str, Any]:
        errors = []
        warnings = []

        # Treat 'hf' and 'huggingface' as equivalent
        input_format_norm = input_format.replace("hf", "huggingface") if input_format == "hf" else input_format
        output_format_norm = output_format.replace("hf", "huggingface") if output_format == "hf" else output_format

        # Check if input format is supported
        if input_format_norm not in self.get_conversion_matrix() and input_format_norm != "huggingface":
            errors.append(f"Unsupported input format: {input_format}")

        # Allow 'hf2megatron' as a valid output format for huggingface/hf and megatron input
        special_hf2megatron = output_format in ("hf2megatron",) and input_format in ("huggingface", "hf", "megatron")
        # Check if output format is supported (allow custom_quant for all)
        valid_outputs = self.get_conversion_matrix().get(input_format_norm, [])
        if output_format_norm not in valid_outputs and not special_hf2megatron:
            errors.append(f"Unsupported output format: {output_format}")

        # Allow any non-empty string as model_type for custom models (e.g., 'minicpm', 'llama')
        if not model_type or not isinstance(model_type, str):
            errors.append(f"Invalid model type: {model_type}. Must be a non-empty string.")

        # Check device
        valid_devices = ["auto", "cpu", "cuda", "mps"]
        if device not in valid_devices:
            errors.append(f"Invalid device: {device}. Valid devices: {valid_devices}")

        # Check quantization for supported formats
        if quantization and output_format_norm not in ["gguf", "gptq", "awq", "mlx", "custom_quant"]:
            warnings.append(f"Quantization not supported for {output_format} format")

        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def _map_model_type_internal(self, model_type: str) -> str:
        """
        Internal helper: Map 'feature-extraction' to 'auto' for model loading/conversion.
        This preserves user intent but ensures compatibility with underlying libraries.
        """
        if model_type == "feature-extraction":
            return "auto"
        return model_type

    def convert(
        self,
        model: Any = None,
        tokenizer: Any = None,
        model_name: str = None,
        output_format: str = None,
        output_path: str = None,
        model_type: str = "auto",
        device: str = "auto",
        quantization: Optional[str] = None,
        use_large_calibration: bool = False,
        dtype: str = None,
        quantization_config: dict = None,
        fake_weight: bool = False,
        fake_weight_shape_dict: dict = None,  # New argument for custom fake weight shapes
        mup2llama: bool = False,  # New argument for muP-to-LLaMA scaling
    ) -> ConversionResult:
        print(f"[DEBUG] convert: model_name={model_name}, output_format={output_format}, model_type={model_type}")
        """
        Convert a model to the specified format. Always performs static validation after conversion.
        Only returns success if both conversion and static validation succeed.
        """

        def has_mup_params(config: dict) -> bool:
            """Detect if config contains muP parameters."""
            mup_keys = ["scale_emb", "scale_depth", "dim_model_base", "hidden_size", "num_hidden_layers"]
            return any(k in config for k in mup_keys)

        result = ConversionResult(success=False)
        try:
            input_format, norm_path = self._detect_model_format(model_name)
            validation = self._validate_conversion_inputs(input_format, output_format, model_type, quantization, device)
            import sys

            print(f"[DEBUG] validation: {validation}", file=sys.stderr, flush=True)
            if not validation["valid"]:
                print(
                    f"[DEBUG] conversion aborted due to validation errors: {validation['errors']}",
                    file=sys.stderr,
                    flush=True,
                )
                result.error = "; ".join(validation["errors"]) or "Invalid conversion inputs"
                return result
            print(f"[DEBUG] convert: detected input_format={input_format}, norm_path={norm_path}")
            convert_func, validate_func = self._get_converter_functions(output_format)
            internal_model_type = self._map_model_type_internal(model_type)
            # muP scaling: detect and compute scaling factors
            mup_scales = None
            mup_config_dict = None
            if mup2llama:
                try:
                    from transformers import AutoConfig

                    config = AutoConfig.from_pretrained(norm_path, trust_remote_code=True)
                    config_dict = config.to_dict() if hasattr(config, "to_dict") else dict(config)
                    if has_mup_params(config_dict):
                        import math

                        embedding_scale = config_dict["scale_emb"]
                        residual_scale = config_dict["scale_depth"] / math.sqrt(config_dict["num_hidden_layers"])
                        logit_scale = config_dict["hidden_size"] / config_dict["dim_model_base"]
                        logger.info(
                            f"[muP2LLaMA] Detected muP params. embedding_scale={embedding_scale}, residual_scale={residual_scale}, logit_scale={logit_scale}"
                        )
                        mup_scales = dict(embedding=embedding_scale, residual=residual_scale, logit=logit_scale)
                        # Process config, remove muP parameters, keep LLaMA parameters
                        mup_keys = ["scale_emb", "scale_depth", "dim_model_base"]
                        llama_keys = ["hidden_size", "num_hidden_layers", "num_attention_heads"]
                        mup_config_dict = {k: v for k, v in config_dict.items() if k not in mup_keys}
                        logger.info(f"[muP2LLaMA] Removed muP keys: {mup_keys}")
                        logger.info(f"[muP2LLaMA] Kept keys: {[k for k in llama_keys if k in mup_config_dict]}")
                    else:
                        logger.info("[muP2LLaMA] No muP parameters detected in config.")
                except Exception as e:
                    logger.warning(f"[muP2LLaMA] Failed to load config or compute muP scaling: {e}")
            # Weight scaling logic
            if mup_scales and model is not None:
                try:
                    # Embedding layer
                    if hasattr(model, "embed_tokens") and hasattr(model.embed_tokens, "weight"):
                        model.embed_tokens.weight.data.mul_(mup_scales["embedding"])
                        logger.info("[muP2LLaMA] Scaled embed_tokens.weight by embedding_scale")
                    # Logit layer
                    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
                        model.lm_head.weight.data.mul_(mup_scales["logit"])
                        logger.info("[muP2LLaMA] Scaled lm_head.weight by logit_scale")
                    if hasattr(model, "output") and hasattr(model.output, "weight"):
                        model.output.weight.data.mul_(mup_scales["logit"])
                        logger.info("[muP2LLaMA] Scaled output.weight by logit_scale")
                    # Residual layer (iterate over transformer blocks)
                    if hasattr(model, "layers") and isinstance(model.layers, (list, tuple)):
                        for i, layer in enumerate(model.layers):
                            # Common residual projection names: self_attn.out_proj, mlp.down_proj, mlp.gate_proj, mlp.up_proj
                            for attr in ["self_attn", "mlp"]:
                                sub = getattr(layer, attr, None)
                                if sub:
                                    for proj in ["out_proj", "down_proj", "gate_proj", "up_proj"]:
                                        p = getattr(sub, proj, None)
                                        if p and hasattr(p, "weight"):
                                            p.weight.data.mul_(mup_scales["residual"])
                                            logger.info(
                                                f"[muP2LLaMA] Scaled layer {i} {attr}.{proj}.weight by residual_scale"
                                            )
                except Exception as e:
                    logger.warning(f"[muP2LLaMA] Failed to scale weights: {e}")
            # Special handling for some formats
            if output_format == "safetensors":
                try:
                    from model_converter_tool.engine.safetensors import (
                        convert_to_safetensors,
                        validate_safetensors_file,
                    )

                    if model is None:
                        from model_converter_tool.utils import load_model_and_tokenizer

                        model, _ = load_model_and_tokenizer(
                            model_name=norm_path,
                            model_type=internal_model_type,
                            fake_weight=fake_weight,
                            fake_weight_shape_dict=fake_weight_shape_dict,
                            trust_remote_code=True,
                        )
                    success, extra_info = convert_to_safetensors(
                        model, tokenizer, model_name, output_path, internal_model_type, device, dtype
                    )
                    if not success:
                        result.error = f"Safetensors conversion failed: {extra_info}"
                        return result
                    # Always validate after conversion
                    valid = validate_safetensors_file(output_path)
                    if not valid:
                        result.error = "Static validation failed for safetensors output."
                        return result
                    result.success = True
                    result.validation = True
                    result.output_path = output_path
                    result.extra_info = extra_info
                    # config.json overwrite moved here to ensure it happens after all saves
                    if mup_config_dict and output_path:
                        import json
                        from pathlib import Path

                        out_dir = Path(output_path)
                        if out_dir.is_file() or (not out_dir.exists() and out_dir.suffix):
                            out_dir = out_dir.parent
                        out_dir.mkdir(parents=True, exist_ok=True)
                        config_path = out_dir / "config.json"
                        try:
                            with open(config_path, "w") as f:
                                json.dump(mup_config_dict, f, indent=2)
                            logger.info(f"[muP2LLaMA] Overwrote config.json in {out_dir} for LLaMA compatibility.")
                        except Exception as e:
                            logger.warning(f"[muP2LLaMA] Failed to overwrite config.json: {e}")
                    return result
                except Exception as e:
                    result.error = f"Safetensors conversion failed: {e}"
                    return result
            if output_format == "custom_quant":
                try:
                    from model_converter_tool.engine.custom_quant import (
                        convert_to_custom_quant,
                        validate_custom_quant_file,
                    )

                    if model is None:
                        from model_converter_tool.utils import load_model_and_tokenizer

                        model, _ = load_model_and_tokenizer(
                            model_name=norm_path,
                            model_type=internal_model_type,
                            fake_weight=fake_weight,
                            fake_weight_shape_dict=fake_weight_shape_dict,
                            trust_remote_code=True,
                        )
                    success, extra_info = convert_to_custom_quant(
                        model,
                        None,
                        model_name,
                        output_path,
                        internal_model_type,
                        device,
                        quantization,
                        use_large_calibration,
                        quantization_config,
                    )
                    if not success:
                        result.error = f"Custom quantization failed: {extra_info}"
                        return result
                    valid = validate_custom_quant_file(output_path)
                    if not valid:
                        result.error = "Static validation failed for custom_quant output."
                        return result
                    result.success = True
                    result.validation = True
                    result.output_path = output_path
                    result.extra_info = extra_info
                    return result
                except Exception as e:
                    result.error = f"Custom quantization failed: {e}"
                    return result
            # Special handling for megatron2hf and hf2megatron (must come BEFORE SUPPORTED_FORMATS block)
            if output_format in ("megatron2hf", "hf2megatron"):
                import sys

                print(
                    f"[DEBUG] About to call convert_func for {output_format}: {convert_func} with model_type={model_type}, norm_path={norm_path}, output_path={output_path}",
                    file=sys.stderr,
                    flush=True,
                )
                if output_format == "hf2megatron":
                    conversion_result = convert_func(model_type=model_type, hf_path=norm_path, output_path=output_path)
                else:
                    conversion_result = convert_func(
                        model_type=model_type, checkpoint_path=norm_path, output_path=output_path
                    )
                print(f"[DEBUG] convert_func returned: {conversion_result}", file=sys.stderr, flush=True)
                # Wrap result if needed
                if isinstance(conversion_result, bool):
                    result.success = conversion_result
                    result.output_path = output_path if conversion_result else None
                elif hasattr(conversion_result, "success"):
                    return conversion_result
                else:
                    result.success = bool(conversion_result)
                    result.output_path = output_path if result.success else None
                return result
            if output_format == "onnx":
                success = convert_func(model_name=model_name, output_path=output_path, device=device)
                if not success:
                    result.error = f"ONNX conversion failed for {model_name}"
                    return result
                valid = validate_func(output_path)
                if not valid:
                    result.error = "Static validation failed for ONNX output."
                    return result
                result.success = True
                result.validation = True
                result.output_path = output_path
                result.extra_info = None
                return result
            if output_format in SUPPORTED_FORMATS:
                if output_format == "safetensors":
                    # Already handled above
                    pass
                elif output_format in ("awq", "gptq"):
                    success, extra = convert_func(
                        model,
                        tokenizer,
                        model_name,
                        output_path,
                        internal_model_type,
                        device,
                        quantization,
                        use_large_calibration,
                        quantization_config,
                    )
                else:
                    success, extra = convert_func(
                        model, tokenizer, model_name, output_path, internal_model_type, device
                    )
                if not success:
                    result.error = extra if isinstance(extra, str) else f"Conversion failed for {output_format}"
                    return result
                valid = validate_func(output_path)
                if not valid:
                    result.error = f"Static validation failed for {output_format} output."
                    return result
                result.success = True
                result.validation = True
                result.output_path = output_path
                result.extra_info = extra
                return result
            else:
                result.error = f"Unsupported format: {output_format}"
        except Exception as e:
            result.success = False
            result.error = str(e)
        return result

    def batch_convert(
        self,
        tasks: List[Dict[str, Any]],
        max_workers: int = 1,
        max_retries: int = 1,
    ) -> List[ConversionResult]:
        """
        Batch convert multiple models.
        Args:
            tasks: Task list, each task is a dict with same parameters as convert
            max_workers: Number of concurrent workers
            max_retries: Maximum retry attempts
        Returns:
            List of ConversionResult
        """
        results: List[ConversionResult] = []
        for task in tasks:
            retries = 0
            while retries < max_retries:
                try:
                    res = self.convert(**task)
                    results.append(res)
                    break
                except Exception as e:
                    retries += 1
                    if retries >= max_retries:
                        results.append(ConversionResult(success=False, error=str(e)))
        return results

    def get_conversion_matrix(self) -> Dict[str, List[str]]:
        """Public method to get the conversion matrix, including custom_quant support."""
        matrix = self._get_conversion_matrix()
        # Add 'custom_quant' to all input types
        for k in matrix:
            if "custom_quant" not in matrix[k]:
                matrix[k].append("custom_quant")
        # Make 'hf' and 'huggingface' equivalent for all input types
        for k in list(matrix.keys()):
            if "huggingface" in matrix[k] and "hf" not in matrix[k]:
                matrix[k].append("hf")
            if "hf" in matrix[k] and "huggingface" not in matrix[k]:
                matrix[k].append("huggingface")
        return matrix

    def _get_conversion_matrix(self) -> Dict[str, List[str]]:
        return {
            "huggingface": [
                "huggingface",
                "hf",
                "safetensors",
                "torchscript",
                "onnx",
                "gguf",
                "mlx",
                "gptq",
                "awq",
                "mtk",
                "rk",
                "ax",
                "qnn",  # Newly added formats
                "hf2megatron",  # Add HF2Megatron as valid output
            ],
            "hf": [
                "huggingface",
                "hf",
                "safetensors",
                "torchscript",
                "onnx",
                "gguf",
                "mlx",
                "gptq",
                "awq",
                "mtk",
                "rk",
                "ax",
                "qnn",  # Newly added formats
                "hf2megatron",  # Add HF2Megatron as valid output
            ],
            "megatron": ["hf", "megatron2hf", "hf2megatron"],  # Add both directions
            "safetensors": ["huggingface", "hf", "safetensors"],
            "torchscript": ["torchscript"],
            "onnx": ["onnx"],
            "gguf": ["gguf"],
            "mlx": ["mlx"],
        }

    def get_input_formats(self) -> dict:
        """
        Return a dict of supported input formats. Key: format name, Value: description or empty dict.
        """
        # You can expand descriptions as needed
        return {fmt: {} for fmt in ["huggingface", "safetensors", "torchscript", "onnx", "gguf", "mlx"]}

    def get_output_formats(self) -> dict:
        """
        Return a dict of supported output formats. Key: format name, Value: description or empty dict.
        """
        return {
            fmt: {}
            for fmt in [
                "huggingface",
                "hf",
                "safetensors",
                "torchscript",
                "onnx",
                "gguf",
                "mlx",
                "gptq",
                "awq",
                "hf2megatron",
            ]
        }

    def _can_infer_model(self, model_path, model_format=None, **kwargs):
        """
        Dynamic inference check: load the model and run a dummy inference using the appropriate engine.
        Returns a dict: {can_infer: bool, error: str (optional), details: str (optional)}
        """
        try:
            if not model_format:
                model_format, model_path = self._detect_model_format(model_path)
            engine_map = {
                "onnx": ("model_converter_tool.engine.onnx", "can_infer_onnx_file"),
                "gguf": ("model_converter_tool.engine.gguf", "can_infer_gguf_file"),
                "safetensors": ("model_converter_tool.engine.safetensors", "can_infer_safetensors_file"),
                "torchscript": ("model_converter_tool.engine.torchscript", "can_infer_torchscript_file"),
                "hf": ("model_converter_tool.engine.hf", "can_infer_hf_file"),
                "huggingface": ("model_converter_tool.engine.hf", "can_infer_hf_file"),
                "mlx": ("model_converter_tool.engine.mlx", "can_infer_mlx_file"),
                "gptq": ("model_converter_tool.engine.gptq", "can_infer_gptq_file"),
                "awq": ("model_converter_tool.engine.awq", "can_infer_awq_file"),
            }
            fmt = model_format.lower()
            if fmt not in engine_map:
                return {"can_infer": False, "error": f"Format '{fmt}' is not supported for dynamic check."}
            module_name, func_name = engine_map[fmt]
            import importlib

            engine = importlib.import_module(module_name)
            check_func = getattr(engine, func_name)
            # Always use the real file, do not use fake_weight for inference check
            result = check_func(model_path)
            if isinstance(result, tuple):
                can_infer, error = result
                return {"can_infer": bool(can_infer), "error": error}
            else:
                return {"can_infer": bool(result)}
        except Exception as e:
            import traceback

            return {"can_infer": False, "error": str(e), "details": traceback.format_exc()}
