"""
Model Converter Tool - API First Design

This module provides the core API interface for the model converter tool.
- Only exposes API methods that correspond to CLI commands
- Separates static validation and dynamic check
- Centralizes engine-level validate/can_infer/convert calls
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from enum import Enum
import logging

from .converter import ModelConverter, ConversionResult

logger = logging.getLogger(__name__)

@dataclass
class ConversionPlan:
    model_path: str
    output_format: str
    output_path: str
    model_type: str = "auto"
    device: str = "auto"
    quantization: Optional[str] = None
    use_large_calibration: bool = False
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    estimated_size: Optional[str] = None
    estimated_memory: Optional[str] = None
    estimated_time: Optional[str] = None
    quantization_config: Optional[dict] = None

class ModelConverterAPI:
    def __init__(self, workspace_path: Optional[Path] = None):
        self.workspace_path = workspace_path or Path.cwd()
        self.converter = ModelConverter()

    def detect_model(self, model_path: str) -> Dict[str, Any]:
        """
        Detect the model format and return detailed information.
        """
        try:
            format_name, normalized_path = self.converter._detect_model_format(model_path)
            metadata = self._get_model_metadata(model_path, format_name)
            return {
                "format": format_name,
                "path": normalized_path,
                "metadata": metadata,
                "supported_outputs": self._get_supported_outputs(format_name),
                "detection_confidence": "high" if format_name != "unknown" else "low"
            }
        except Exception as e:
            return {
                "format": "unknown",
                "path": model_path,
                "error": str(e),
                "supported_outputs": [],
                "detection_confidence": "none"
            }

    def validate_conversion(self, model_path: str, output_format: str, **kwargs) -> Dict[str, Any]:
        """
        Static validation: check if input/output formats and parameters are valid and convertible.
        """
        format_info = self.detect_model(model_path)
        input_format = format_info["format"]
        validation = self.converter._validate_conversion_inputs(
            input_format, output_format,
            kwargs.get("model_type", "auto"),
            kwargs.get("quantization", ""),
            kwargs.get("device", "auto")
        )
        plan = ConversionPlan(
            model_path=model_path,
            output_format=output_format,
            output_path=kwargs.get("output_path", ""),
            model_type=kwargs.get("model_type", "auto"),
            device=kwargs.get("device", "auto"),
            quantization=kwargs.get("quantization"),
            use_large_calibration=kwargs.get("use_large_calibration", False),
            is_valid=validation["valid"],
            errors=validation["errors"],
            warnings=validation["warnings"],
            quantization_config=kwargs.get("quantization_config")
        )
        if plan.is_valid:
            estimates = self._estimate_conversion(plan)
            plan.estimated_size = estimates.get("size")
            plan.estimated_memory = estimates.get("memory")
            plan.estimated_time = estimates.get("time")
        return {
            "valid": plan.is_valid,
            "plan": plan,
            "format_info": format_info,
            "validation": validation
        }

    def check_model(self, model_path: str, model_format: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Dynamic check: actually load the model and try inference, return if inference is possible.
        """
        try:
            if not model_format:
                format_name, norm_path = self.converter._detect_model_format(model_path)
                model_format = format_name
                model_path = norm_path
            else:
                model_format = model_format.lower()
            # Dynamic inference check, call engine's can_infer interface
            result = self.converter._can_infer_model(model_path, model_format, **kwargs)
            return {
                "can_infer": result["can_infer"],
                "error": result.get("error"),
                "details": result.get("details")
            }
        except Exception as e:
            return {"can_infer": False, "error": str(e)}

    def convert_model(self, model_path: str, output_format: str, output_path: str, fake_weight_shape_dict: dict = None, mup2llama: bool = False, **kwargs) -> ConversionResult:
        """
        Execute model conversion.
        """
        # Static validation first
        validation = self.validate_conversion(model_path, output_format, output_path=output_path, **kwargs)
        plan = validation["plan"]
        if not plan.is_valid:
            return ConversionResult(success=False, error="Invalid conversion plan", extra_info={"plan": plan})
        try:
            # Remove main arguments from kwargs to avoid multiple values error
            filtered_kwargs = dict(kwargs)
            for k in ["model_type", "output_format", "output_path", "device", "quantization", "use_large_calibration", "quantization_config", "model_name"]:
                filtered_kwargs.pop(k, None)
            result = self.converter.convert(
                model_name=plan.model_path,
                output_format=plan.output_format,
                output_path=plan.output_path,
                model_type=plan.model_type,
                device=plan.device,
                quantization=plan.quantization,
                use_large_calibration=plan.use_large_calibration,
                quantization_config=getattr(plan, 'quantization_config', None),
                fake_weight_shape_dict=fake_weight_shape_dict,  # Pass the custom shape dict
                mup2llama=mup2llama,
                **filtered_kwargs
            )
            return result
        except Exception as e:
            return ConversionResult(success=False, error=str(e), extra_info={"plan": plan})

    def inspect_model(self, model_path: str) -> Dict[str, Any]:
        """
        Inspect model details.
        """
        return self.detect_model(model_path)

    def list_supported(self, full: bool = True) -> Dict[str, Any]:
        """
        Get supported formats and conversion matrix (if full=True), or just input/output formats (if full=False).
        """
        if full:
            return {
                "input_formats": self._get_input_formats(),
                "output_formats": self._get_output_formats(),
                "conversion_matrix": self._get_conversion_matrix()
            }
        else:
            return {
                "input_formats": self._get_input_formats(),
                "output_formats": self._get_output_formats(),
            }

    def manage_config(self, action: str = "show", key: str = None, value: str = None):
        from model_converter_tool.config import ConfigManager
        mgr = ConfigManager()
        if action == "show":
            return mgr.all()
        elif action == "get" and key:
            return {key: mgr.get(key)}
        elif action == "set" and key:
            mgr.set(key, value)
            return {key: mgr.get(key)}
        elif action == "list_presets":
            return mgr.list_presets()
        else:
            return {"error": "Unknown config action"}

    def get_history(self) -> Dict[str, Any]:
        from model_converter_tool.core.history import get_history
        return get_history()

    # --- Private helper methods ---
    def _get_model_metadata(self, model_path: str, format_name: str) -> Dict[str, Any]:
        return {"format": format_name, "path": model_path, "size": "unknown"}

    def _get_supported_outputs(self, input_format: str) -> List[str]:
        return self.converter.get_conversion_matrix().get(input_format, [])

    def _get_conversion_matrix(self) -> Dict[str, List[str]]:
        return self.converter.get_conversion_matrix()

    def _estimate_conversion(self, plan: ConversionPlan) -> Dict[str, str]:
        import os
        try:
            if os.path.exists(plan.model_path):
                model_size = os.path.getsize(plan.model_path)
                size_note = ""
            else:
                lower_name = plan.model_path.lower()
                if "bert-base" in lower_name:
                    model_size = 420 * 1024 ** 2
                    size_note = " (estimated for BERT-base)"
                elif "llama" in lower_name:
                    model_size = 3 * 1024 ** 3
                    size_note = " (estimated for LLaMA)"
                elif "gpt2" in lower_name:
                    model_size = 500 * 1024 ** 2
                    size_note = " (estimated for GPT-2)"
                else:
                    model_size = 1 * 1024 ** 3
                    size_note = " (default estimate)"
            estimated_size = f"{model_size / (1024**2):.2f} MB{size_note}"
            estimated_memory = f"{model_size * 3 / (1024**3):.2f} GB{size_note}"
            gb = model_size / (1024**3)
            min_time = int(gb * 30)
            max_time = int(gb * 60)
            estimated_time = f"{min_time}~{max_time} s{size_note}"
            return {"size": estimated_size, "memory": estimated_memory, "time": estimated_time}
        except Exception:
            return {"size": "unknown", "memory": "unknown", "time": "unknown"}

    def _get_input_formats(self) -> Dict[str, Any]:
        return self.converter.get_input_formats()

    def _get_output_formats(self) -> Dict[str, Any]:
        return self.converter.get_output_formats() 