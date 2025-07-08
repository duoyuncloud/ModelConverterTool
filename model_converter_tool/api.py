"""
Model Converter Tool - API First Design

This module provides the core API interface for the model converter tool.
Following API First principles:
- Clear, consistent API design
- Rich return types with structured data
- Proper error handling and validation
- CLI and other clients can build on top of this API
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from enum import Enum
import logging

from .converter import ModelConverter, ConversionResult

logger = logging.getLogger(__name__)


class ConversionStatus(Enum):
    """Conversion status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ConversionPlan:
    """Conversion plan with detailed information"""
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


@dataclass
class ConversionTask:
    """A conversion task with full context"""
    id: str
    plan: ConversionPlan
    status: ConversionStatus = ConversionStatus.PENDING
    result: Optional[ConversionResult] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    progress: float = 0.0
    error_message: Optional[str] = None


@dataclass
class WorkspaceStatus:
    """Workspace status information"""
    workspace_path: Path
    active_tasks: List[ConversionTask] = field(default_factory=list)
    completed_tasks: List[ConversionTask] = field(default_factory=list)
    failed_tasks: List[ConversionTask] = field(default_factory=list)
    cache_size: Optional[str] = None
    available_formats: Dict[str, List[str]] = field(default_factory=dict)


class ModelConverterAPI:
    """
    Core API for model conversion operations.
    
    This class provides a clean, consistent API that CLI and other clients
    can build upon. It handles:
    - Model format detection and validation
    - Conversion planning and execution
    - Progress tracking and status management
    - Error handling and recovery
    """
    
    def __init__(self, workspace_path: Optional[Path] = None):
        self.workspace_path = workspace_path or Path.cwd()
        self.converter = ModelConverter()
        self._active_tasks: Dict[str, ConversionTask] = {}
        
    def detect_model(self, model_path: str) -> Dict[str, Any]:
        """
        Detect model format and provide detailed information.
        
        Args:
            model_path: Path to model or model name
            
        Returns:
            Dictionary with format information and metadata
        """
        try:
            format_name, normalized_path = self.converter._detect_model_format(model_path)
            
            # Get additional metadata
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
    
    def validate_conversion(
        self, 
        model_path: str, 
        output_format: str, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Validate a conversion request and return detailed validation results.
        
        Args:
            model_path: Input model path
            output_format: Target format
            **kwargs: Additional conversion parameters
            
        Returns:
            Validation result with detailed information
        """
        # Detect input format
        format_info = self.detect_model(model_path)
        input_format = format_info["format"]
        
        # Validate conversion
        validation = self.converter._validate_conversion_inputs(
            input_format, output_format, 
            kwargs.get("model_type", "auto"),
            kwargs.get("quantization", ""),
            kwargs.get("device", "auto")
        )
        
        # Create conversion plan
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
            warnings=validation["warnings"]
        )
        
        # Estimate size and time if valid
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
    
    def plan_conversion(
        self, 
        model_path: str, 
        output_format: str, 
        output_path: str,
        **kwargs
    ) -> ConversionPlan:
        """
        Create a conversion plan without executing it.
        
        Args:
            model_path: Input model path
            output_format: Target format
            output_path: Output path
            **kwargs: Additional parameters
            
        Returns:
            ConversionPlan with all details
        """
        validation = self.validate_conversion(
            model_path, output_format, output_path=output_path, **kwargs
        )
        
        plan = validation["plan"]
        plan.output_path = output_path
        
        return plan
    
    def execute_conversion(
        self, 
        plan: ConversionPlan,
        track_progress: bool = True
    ) -> ConversionResult:
        """
        Execute a conversion plan.
        
        Args:
            plan: ConversionPlan to execute
            track_progress: Whether to track progress
            
        Returns:
            ConversionResult with execution details
        """
        if not plan.is_valid:
            return ConversionResult(
                success=False,
                error="Invalid conversion plan",
                extra_info={"plan": plan}
            )
        
        # Create task for tracking
        task_id = f"{Path(plan.model_path).stem}_{plan.output_format}_{id(plan)}"
        task = ConversionTask(id=task_id, plan=plan)
        
        if track_progress:
            self._active_tasks[task_id] = task
            task.status = ConversionStatus.RUNNING
        
        try:
            # Execute conversion
            result = self.converter.convert(
                model_name=plan.model_path,
                output_format=plan.output_format,
                output_path=plan.output_path,
                model_type=plan.model_type,
                device=plan.device,
                quantization=plan.quantization,
                use_large_calibration=plan.use_large_calibration
            )
            
            if track_progress:
                task.result = result
                task.status = ConversionStatus.SUCCESS if result.success else ConversionStatus.FAILED
                task.end_time = self._get_current_time()
            
            return result
            
        except Exception as e:
            error_result = ConversionResult(
                success=False,
                error=str(e),
                extra_info={"plan": plan}
            )
            
            if track_progress:
                task.result = error_result
                task.status = ConversionStatus.FAILED
                task.error_message = str(e)
                task.end_time = self._get_current_time()
            
            return error_result
    
    def get_workspace_status(self) -> WorkspaceStatus:
        """
        Get current workspace status.
        
        Returns:
            WorkspaceStatus with current state
        """
        active_tasks = list(self._active_tasks.values())
        completed_tasks = [t for t in active_tasks if t.status == ConversionStatus.SUCCESS]
        failed_tasks = [t for t in active_tasks if t.status == ConversionStatus.FAILED]
        running_tasks = [t for t in active_tasks if t.status == ConversionStatus.RUNNING]
        
        return WorkspaceStatus(
            workspace_path=self.workspace_path,
            active_tasks=running_tasks,
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            available_formats=self._get_available_formats()
        )
    
    def get_supported_formats(self) -> Dict[str, Any]:
        """
        Get supported format information.
        
        Returns:
            Dictionary with format support matrix
        """
        return {
            "input_formats": self._get_input_formats(),
            "output_formats": self._get_output_formats(),
            "conversion_matrix": self._get_conversion_matrix()
        }
    
    # Private helper methods
    
    def _get_model_metadata(self, model_path: str, format_name: str) -> Dict[str, Any]:
        """Get model metadata based on format"""
        return {
            "format": format_name,
            "path": model_path,
            "size": "unknown"
        }
    
    def _get_supported_outputs(self, input_format: str) -> List[str]:
        """Get supported output formats for input format"""
        format_matrix = {
            "huggingface": ["onnx", "gguf", "torchscript", "fp16", "gptq", "awq", "safetensors", "mlx"],
            "safetensors": ["onnx", "gguf", "torchscript", "fp16", "gptq", "awq", "safetensors", "mlx"],
            "torchscript": ["onnx", "torchscript"],
            "onnx": ["onnx"],
            "gguf": ["gguf"],
            "mlx": ["mlx"]
        }
        return format_matrix.get(input_format, [])
    
    def _estimate_conversion(self, plan: ConversionPlan) -> Dict[str, str]:
        """Estimate conversion size, memory, and time"""
        import os
        try:
            # 判断是否为本地文件
            if os.path.exists(plan.model_path):
                model_size = os.path.getsize(plan.model_path)
                size_note = ""
            else:
                # 针对常见模型名给出典型大小
                lower_name = plan.model_path.lower()
                if "bert-base" in lower_name:
                    model_size = 420 * 1024 ** 2  # 420MB
                    size_note = " (estimated for BERT-base)"
                elif "llama" in lower_name:
                    model_size = 3 * 1024 ** 3  # 3GB
                    size_note = " (estimated for LLaMA)"
                elif "gpt2" in lower_name:
                    model_size = 500 * 1024 ** 2  # 500MB
                    size_note = " (estimated for GPT-2)"
                else:
                    model_size = 1 * 1024 ** 3  # 默认1GB
                    size_note = " (default estimate)"
            estimated_size = f"{model_size / (1024**2):.2f} MB{size_note}"
            estimated_memory = f"{model_size * 3 / (1024**3):.2f} GB{size_note}"
            gb = model_size / (1024**3)
            min_time = int(gb * 30)
            max_time = int(gb * 60)
            estimated_time = f"{min_time}~{max_time} s{size_note}"
            return {
                "size": estimated_size,
                "memory": estimated_memory,
                "time": estimated_time
            }
        except Exception:
            return {
                "size": "unknown",
                "memory": "unknown",
                "time": "unknown"
            }
    
    def _get_current_time(self) -> float:
        """Get current timestamp"""
        import time
        return time.time()
    
    def _get_available_formats(self) -> Dict[str, List[str]]:
        """Get available formats in current environment"""
        return {
            "input": ["huggingface", "safetensors", "torchscript", "onnx", "gguf"],
            "output": ["onnx", "gguf", "torchscript", "fp16", "gptq", "awq", "safetensors", "mlx"]
        }
    
    def _get_input_formats(self) -> Dict[str, Any]:
        """Get input format information"""
        return {
            "huggingface": {
                "description": "Hugging Face Transformers format",
                "extensions": [".py", "config.json", "pytorch_model.bin", "*.safetensors"],
                "detection": "Directory with config.json and model files"
            },
            "safetensors": {
                "description": "SafeTensors format",
                "extensions": [".safetensors"],
                "detection": "SafeTensors file"
            },
            "torchscript": {
                "description": "TorchScript format",
                "extensions": [".pt", ".pth"],
                "detection": "TorchScript file"
            },
            "onnx": {
                "description": "ONNX format",
                "extensions": [".onnx"],
                "detection": "ONNX file"
            },
            "gguf": {
                "description": "GGUF format",
                "extensions": [".gguf"],
                "detection": "GGUF file"
            }
        }
    
    def _get_output_formats(self) -> Dict[str, Any]:
        """Get output format information"""
        return {
            "onnx": {
                "description": "ONNX format - Cross-platform inference standard",
                "extensions": [".onnx"],
                "use_cases": ["Cross-platform deployment", "TensorRT optimization", "Mobile inference"],
                "quantization": False
            },
            "gguf": {
                "description": "GGUF format - llama.cpp optimized",
                "extensions": [".gguf"],
                "use_cases": ["CPU inference", "Edge devices", "llama.cpp ecosystem"],
                "quantization": True,
                "quantization_options": ["q4_k_m", "q8_0", "q5_k_m", "q4_0", "q4_1"]
            },
            "torchscript": {
                "description": "TorchScript format - PyTorch optimized",
                "extensions": [".pt", ".pth"],
                "use_cases": ["PyTorch ecosystem", "C++ inference", "Mobile"],
                "quantization": False
            },
            "fp16": {
                "description": "FP16 half-precision format",
                "extensions": [".bin", ".safetensors"],
                "use_cases": ["GPU inference", "Memory optimization", "Speed optimization"],
                "quantization": False
            },
            "gptq": {
                "description": "GPTQ quantization format",
                "extensions": [".bin", ".safetensors"],
                "use_cases": ["GPU inference", "High compression", "Maintain accuracy"],
                "quantization": True,
                "quantization_options": ["4bit", "8bit"]
            },
            "awq": {
                "description": "AWQ quantization format",
                "extensions": [".bin", ".safetensors"],
                "use_cases": ["GPU inference", "Activation-aware quantization", "High accuracy"],
                "quantization": True,
                "quantization_options": ["4bit", "8bit"]
            },
            "safetensors": {
                "description": "SafeTensors secure format",
                "extensions": [".safetensors"],
                "use_cases": ["Secure storage", "Fast loading", "Hugging Face ecosystem"],
                "quantization": False
            },
            "mlx": {
                "description": "MLX format - Apple Silicon optimized",
                "extensions": [".mlx"],
                "use_cases": ["Apple Silicon", "macOS optimization", "Local inference"],
                "quantization": True,
                "quantization_options": ["q4_k_m", "q8_0", "q5_k_m"]
            }
        }
    
    def _get_conversion_matrix(self) -> Dict[str, List[str]]:
        """Get conversion compatibility matrix"""
        return {
            "huggingface": ["onnx", "gguf", "torchscript", "fp16", "gptq", "awq", "safetensors", "mlx"],
            "safetensors": ["onnx", "gguf", "torchscript", "fp16", "gptq", "awq", "safetensors", "mlx"],
            "torchscript": ["onnx", "torchscript"],
            "onnx": ["onnx"],
            "gguf": ["gguf"]
        } 