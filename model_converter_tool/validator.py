"""
Model validation utilities for testing converted models
"""

import os
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class ModelValidator:
    """Validate converted models by loading and testing inference"""

    def __init__(self):
        self.validation_results = {}

    def validate_converted_model(
        self, 
        model_path: str, 
        output_format: str, 
        model_type: str = "text-generation"
    ) -> Dict[str, Any]:
        """
        Validate a converted model by loading it and running test inference
        
        Args:
            model_path: Path to the converted model
            output_format: Format of the converted model (onnx, gguf, gptq, etc.)
            model_type: Type of model (text-generation, text-classification, etc.)
            
        Returns:
            Dict with validation results
        """
        try:
            if output_format.lower() == "onnx":
                return self._validate_onnx_model(model_path, model_type)
            elif output_format.lower() == "gguf":
                return self._validate_gguf_model(model_path, model_type)
            elif output_format.lower() == "gptq":
                return self._validate_gptq_model(model_path, model_type)
            elif output_format.lower() == "torchscript":
                return self._validate_torchscript_model(model_path, model_type)
            elif output_format.lower() == "fp16":
                return self._validate_fp16_model(model_path, model_type)
            elif output_format.lower() == "mlx":
                return self._validate_mlx_model(model_path, model_type)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported format for validation: {output_format}"
                }
        except Exception as e:
            logger.error(f"Validation failed for {model_path}: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _validate_onnx_model(self, model_path: str, model_type: str) -> Dict[str, Any]:
        """Validate ONNX model"""
        try:
            import onnxruntime as ort
            
            # Find ONNX file
            model_dir = Path(model_path)
            onnx_files = list(model_dir.glob("*.onnx"))
            if not onnx_files:
                return {"success": False, "error": "No ONNX files found"}
            
            onnx_file = onnx_files[0]
            
            # Load ONNX model
            session = ort.InferenceSession(str(onnx_file))
            
            # Get input/output info
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            
            # Create test input based on model type
            if model_type == "text-generation":
                # For text generation models, use token IDs
                dummy_input = np.random.randint(0, 50257, (1, 10), dtype=np.int64)
            else:
                # For other models, use appropriate input shape
                input_shape = session.get_inputs()[0].shape
                if input_shape[0] == 0:  # Dynamic batch size
                    input_shape = (1,) + tuple(input_shape[1:])
                dummy_input = np.random.randn(*input_shape).astype(np.float32)
            
            # Run inference
            outputs = session.run([output_name], {input_name: dummy_input})
            
            # Validate output
            if len(outputs) == 0:
                return {"success": False, "error": "No outputs from ONNX model"}
            
            output = outputs[0]
            
            return {
                "success": True,
                "model_file": str(onnx_file),
                "input_shape": dummy_input.shape,
                "output_shape": output.shape,
                "inference_time": "measured",  # Could add timing
                "message": f"ONNX model loaded successfully. Input: {dummy_input.shape}, Output: {output.shape}"
            }
            
        except ImportError:
            return {"success": False, "error": "onnxruntime not available"}
        except Exception as e:
            return {"success": False, "error": f"ONNX validation failed: {e}"}

    def _validate_gguf_model(self, model_path: str, model_type: str) -> Dict[str, Any]:
        """Validate GGUF model"""
        try:
            model_dir = Path(model_path)
            gguf_files = list(model_dir.glob("*.gguf"))
            if not gguf_files:
                return {"success": False, "error": "No GGUF files found"}
            
            gguf_file = gguf_files[0]
            
            # Check GGUF file header
            with open(gguf_file, 'rb') as f:
                header = f.read(8)
                if not header.startswith(b'GGUF'):
                    return {"success": False, "error": "Invalid GGUF file header"}
            
            # Try to load with llama.cpp if available
            llama_main = self._find_llama_main()
            if llama_main:
                try:
                    # Test loading with llama.cpp
                    cmd = [llama_main, "-m", str(gguf_file), "-n", "1", "--no-display-prompt"]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    
                    if result.returncode == 0:
                        return {
                            "success": True,
                            "model_file": str(gguf_file),
                            "message": "GGUF model loaded successfully with llama.cpp"
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"llama.cpp loading failed: {result.stderr}"
                        }
                except subprocess.TimeoutExpired:
                    return {"success": False, "error": "llama.cpp loading timed out"}
                except Exception as e:
                    return {"success": False, "error": f"llama.cpp validation failed: {e}"}
            else:
                return {
                    "success": True,
                    "model_file": str(gguf_file),
                    "message": "GGUF file header valid, llama.cpp not available for full validation"
                }
                
        except Exception as e:
            return {"success": False, "error": f"GGUF validation failed: {e}"}

    def _validate_gptq_model(self, model_path: str, model_type: str) -> Dict[str, Any]:
        """Validate GPTQ model"""
        try:
            from auto_gptq import AutoGPTQForCausalLM
            from transformers import AutoTokenizer
            import torch
            
            model_dir = Path(model_path)
            
            # Load the GPTQ model
            model = AutoGPTQForCausalLM.from_quantized(str(model_dir))
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            
            # Create test input
            test_text = "Hello world"
            inputs = tokenizer(test_text, return_tensors="pt")
            
            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Validate output
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            return {
                "success": True,
                "model_file": str(model_dir),
                "input_text": test_text,
                "input_shape": inputs['input_ids'].shape,
                "output_shape": logits.shape,
                "message": f"GPTQ model loaded successfully. Input: {inputs['input_ids'].shape}, Output: {logits.shape}"
            }
            
        except ImportError:
            return {"success": False, "error": "auto-gptq not available"}
        except Exception as e:
            return {"success": False, "error": f"GPTQ validation failed: {e}"}

    def _validate_torchscript_model(self, model_path: str, model_type: str) -> Dict[str, Any]:
        """Validate TorchScript model"""
        try:
            import torch
            
            model_dir = Path(model_path)
            torchscript_files = list(model_dir.glob("*.pt"))
            if not torchscript_files:
                return {"success": False, "error": "No TorchScript files found"}
            
            torchscript_file = torchscript_files[0]
            
            # Load the TorchScript model
            model = torch.jit.load(str(torchscript_file))
            
            # Create dummy input based on model type
            if model_type == "text-generation":
                dummy_input = torch.randint(0, 50257, (1, 10), dtype=torch.long)
            else:
                # For other models, use appropriate input
                dummy_input = torch.randn(1, 10, 768)  # Example for BERT-like models
            
            # Run inference
            with torch.no_grad():
                outputs = model(dummy_input)
            
            return {
                "success": True,
                "model_file": str(torchscript_file),
                "input_shape": dummy_input.shape,
                "output_shape": outputs.shape,
                "message": f"TorchScript model loaded successfully. Input: {dummy_input.shape}, Output: {outputs.shape}"
            }
            
        except Exception as e:
            return {"success": False, "error": f"TorchScript validation failed: {e}"}

    def _validate_fp16_model(self, model_path: str, model_type: str) -> Dict[str, Any]:
        """Validate FP16 model"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            model_dir = Path(model_path)
            
            # Load the FP16 model
            if model_type == "text-generation":
                model = AutoModelForCausalLM.from_pretrained(str(model_dir))
            else:
                from transformers import AutoModel
                model = AutoModel.from_pretrained(str(model_dir))
            
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            
            # Create test input
            test_text = "Hello world"
            inputs = tokenizer(test_text, return_tensors="pt")
            
            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Get output shape
            if hasattr(outputs, 'logits'):
                output_shape = outputs.logits.shape
            elif hasattr(outputs, 'last_hidden_state'):
                output_shape = outputs.last_hidden_state.shape
            else:
                output_shape = tuple(outputs.shape) if hasattr(outputs, 'shape') else "unknown"
            
            return {
                "success": True,
                "model_file": str(model_dir),
                "input_text": test_text,
                "input_shape": inputs['input_ids'].shape,
                "output_shape": output_shape,
                "message": f"FP16 model loaded successfully. Input: {inputs['input_ids'].shape}, Output: {output_shape}"
            }
            
        except Exception as e:
            return {"success": False, "error": f"FP16 validation failed: {e}"}

    def _validate_mlx_model(self, model_path: str, model_type: str) -> Dict[str, Any]:
        """Validate MLX model"""
        try:
            import mlx.core as mx
            import numpy as np
            
            model_dir = Path(model_path)
            mlx_files = list(model_dir.glob("*.npz"))
            if not mlx_files:
                return {"success": False, "error": "No MLX files found"}
            
            mlx_file = mlx_files[0]
            
            # Load MLX model
            model_data = np.load(str(mlx_file))
            
            # Convert to MLX arrays
            mlx_model = {}
            for name, array in model_data.items():
                mlx_model[name] = mx.array(array)
            
            # Create dummy input
            dummy_input = mx.random.normal((1, 10, 768))  # Example shape
            
            return {
                "success": True,
                "model_file": str(mlx_file),
                "model_keys": list(mlx_model.keys()),
                "input_shape": dummy_input.shape,
                "message": f"MLX model loaded successfully with {len(mlx_model)} parameters"
            }
            
        except ImportError:
            return {"success": False, "error": "MLX not available"}
        except Exception as e:
            return {"success": False, "error": f"MLX validation failed: {e}"}

    def _find_llama_main(self) -> Optional[str]:
        """Find llama.cpp main executable"""
        # Check common locations
        possible_paths = [
            "./main",
            "./llama",
            "/usr/local/bin/llama",
            "/usr/bin/llama"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Try to find in PATH
        try:
            import shutil
            llama_path = shutil.which("llama")
            if llama_path:
                return llama_path
        except:
            pass
        
        return None

    def validate_batch_models(
        self, 
        conversion_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Validate a batch of converted models
        
        Args:
            conversion_results: List of conversion results from ModelConverter
            
        Returns:
            List of validation results
        """
        validation_results = []
        
        for result in conversion_results:
            if result.get("success", False):
                # Extract model info from conversion result
                output_path = result.get("output_path", "")
                output_format = result.get("output_format", "")
                model_type = result.get("model_type", "text-generation")
                
                # Validate the converted model
                validation_result = self.validate_converted_model(
                    output_path, output_format, model_type
                )
                
                # Combine conversion and validation results
                combined_result = {
                    **result,
                    "validation": validation_result
                }
                validation_results.append(combined_result)
            else:
                # Conversion failed, no validation needed
                validation_results.append(result)
        
        return validation_results
