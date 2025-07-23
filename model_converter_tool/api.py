"""
Model Converter Tool - API First Design (simplified for DRY/KISS/YAGNI)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
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
    quantization_config: Optional[dict] = None
    use_large_calibration: bool = False
    dtype: Optional[str] = None
    fake_weight: bool = False
    fake_weight_shape_dict: Optional[dict] = None
    mup2llama: bool = False
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ModelConverterAPI:
    def __init__(self, workspace_path: Optional[Path] = None):
        self.workspace_path = workspace_path or Path.cwd()
        self.converter = ModelConverter()

    def detect_model(self, model_path: str) -> Dict[str, Any]:
        try:
            format_name, normalized_path = self.converter._detect_model_format(model_path)
            return {
                "format": format_name,
                "path": normalized_path,
                "supported_outputs": self._get_supported_outputs(format_name),
                "detection_confidence": "high" if format_name != "unknown" else "low",
            }
        except Exception as e:
            return {
                "format": "unknown",
                "path": model_path,
                "error": str(e),
                "supported_outputs": [],
                "detection_confidence": "none",
            }

    def validate_conversion(self, model_path: str, output_format: str, **kwargs) -> Dict[str, Any]:
        format_info = self.detect_model(model_path)
        input_format = format_info["format"]
        validation = self.converter._validate_conversion_inputs(
            input_format,
            output_format,
            kwargs.get("model_type", "auto"),
            kwargs.get("quantization", ""),
            kwargs.get("device", "auto"),
        )
        plan = ConversionPlan(
            model_path=model_path,
            output_format=output_format,
            output_path=kwargs.get("output_path", ""),
            model_type=kwargs.get("model_type", "auto"),
            device=kwargs.get("device", "auto"),
            quantization=kwargs.get("quantization"),
            quantization_config=kwargs.get("quantization_config"),
            use_large_calibration=kwargs.get("use_large_calibration", False),
            dtype=kwargs.get("dtype"),
            fake_weight=kwargs.get("fake_weight", False),
            fake_weight_shape_dict=kwargs.get("fake_weight_shape_dict"),
            mup2llama=kwargs.get("mup2llama", False),
            is_valid=validation["valid"],
            errors=validation["errors"],
            warnings=validation["warnings"],
        )
        return {"valid": plan.is_valid, "plan": plan, "format_info": format_info, "validation": validation}

    def check_model(self, model_path: str, model_format: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        try:
            if not model_format:
                format_name, norm_path = self.converter._detect_model_format(model_path)
                model_format = format_name
                model_path = norm_path
            else:
                model_format = model_format.lower()
            result = self.converter._can_infer_model(model_path, model_format, **kwargs)
            return {"can_infer": result["can_infer"], "error": result.get("error"), "details": result.get("details")}
        except Exception as e:
            return {"can_infer": False, "error": str(e)}

    def convert_model(
        self,
        model_path: str,
        output_format: str,
        output_path: str,
        fake_weight_shape_dict: dict = None,
        mup2llama: bool = False,
        use_large_calibration: bool = False,
        dtype: str = None,
        quantization_config: dict = None,
        fake_weight: bool = False,
        model_type: str = "auto",
        device: str = "auto",
        quantization: str = None,
        **kwargs,
    ) -> ConversionResult:
        validation = self.validate_conversion(
            model_path,
            output_format,
            output_path=output_path,
            model_type=model_type,
            device=device,
            quantization=quantization,
            quantization_config=quantization_config,
            use_large_calibration=use_large_calibration,
            dtype=dtype,
            fake_weight=fake_weight,
            fake_weight_shape_dict=fake_weight_shape_dict,
            mup2llama=mup2llama,
            **kwargs,
        )
        plan = validation["plan"]
        if not plan.is_valid:
            return ConversionResult(success=False, error="Invalid conversion plan", extra_info={"plan": plan})
        try:
            result = self.converter.convert(
                model_name=plan.model_path,
                output_format=plan.output_format,
                output_path=plan.output_path,
                model_type=plan.model_type,
                device=plan.device,
                quantization=plan.quantization,
                quantization_config=plan.quantization_config,
                use_large_calibration=plan.use_large_calibration,
                dtype=plan.dtype,
                fake_weight=plan.fake_weight,
                fake_weight_shape_dict=plan.fake_weight_shape_dict,
                mup2llama=plan.mup2llama,
                **kwargs,
            )
            return result
        except Exception as e:
            return ConversionResult(success=False, error=str(e), extra_info={"plan": plan})

    def inspect_model(self, model_path: str) -> Dict[str, Any]:
        return self.detect_model(model_path)

    def list_supported(self, full: bool = True) -> Dict[str, Any]:
        if full:
            return {
                "input_formats": self._get_input_formats(),
                "output_formats": self._get_output_formats(),
                "conversion_matrix": self._get_conversion_matrix(),
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

    def _get_supported_outputs(self, input_format: str) -> List[str]:
        return self.converter.get_conversion_matrix().get(input_format, [])

    def _get_conversion_matrix(self) -> Dict[str, List[str]]:
        return self.converter.get_conversion_matrix()

    def _get_input_formats(self) -> Dict[str, Any]:
        return self.converter.get_input_formats()

    def _get_output_formats(self) -> Dict[str, Any]:
        return self.converter.get_output_formats()
