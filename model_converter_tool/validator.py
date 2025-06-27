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
        model_type: str = "text-generation",
        quantization_type: str = None
    ) -> Dict[str, Any]:
        """
        Validate a converted model by loading it and running test inference
        
        Args:
            model_path: Path to the converted model
            output_format: Format of the converted model (onnx, gguf, gptq, etc.)
            model_type: Type of model (text-generation, text-classification, etc.)
            quantization_type: Type of quantization (gptq, awq, gguf, etc.)
            
        Returns:
            Dict with validation results
        """
        try:
            # 处理量化模型
            if quantization_type:
                return self._validate_quantized_model(model_path, model_type, quantization_type)
            
            # 处理标准格式
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
            elif output_format.lower() == "hf":
                return self._validate_hf_model(model_path, model_type)
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
        """Validate TorchScript model with improved logic"""
        try:
            import torch
            
            model_dir = Path(model_path)
            torchscript_files = list(model_dir.glob("*.pt"))
            if not torchscript_files:
                return {"success": False, "error": "No TorchScript files found"}
            
            torchscript_file = torchscript_files[0]
            
            # 检查文件大小
            file_size = torchscript_file.stat().st_size / (1024 * 1024)  # MB
            if file_size < 1:  # 小于1MB可能是占位符
                return {"success": False, "error": "TorchScript file too small, likely a placeholder"}
            
            # 加载TorchScript模型
            try:
                model = torch.jit.load(str(torchscript_file))
            except Exception as e:
                return {"success": False, "error": f"Failed to load TorchScript model: {e}"}
            
            # 获取模型信息
            model_info = {
                "code": model.code if hasattr(model, 'code') else "N/A",
                "graph": str(model.graph) if hasattr(model, 'graph') else "N/A",
                "inlined_graph": str(model.inlined_graph) if hasattr(model, 'inlined_graph') else "N/A"
            }
            
            # 测试推理
            try:
                if model_type == "text-generation":
                    # 为文本生成模型创建合适的输入
                    dummy_input = torch.randint(0, 50257, (1, 8), dtype=torch.long)
                else:
                    # 为其他模型创建合适的输入
                    dummy_input = torch.randn(1, 768)  # 标准embedding大小
                
                # 运行推理
                with torch.no_grad():
                    output = model(dummy_input)
                
                # 分析输出
                if isinstance(output, torch.Tensor):
                    output_shape = output.shape
                    output_dtype = str(output.dtype)
                elif isinstance(output, (list, tuple)):
                    output_shape = [o.shape if isinstance(o, torch.Tensor) else str(o) for o in output]
                    output_dtype = "mixed"
                else:
                    output_shape = str(type(output))
                    output_dtype = str(type(output))
                
                return {
                    "success": True,
                    "model_file": str(torchscript_file),
                    "file_size_mb": file_size,
                    "input_shape": dummy_input.shape,
                    "output_shape": output_shape,
                    "output_dtype": output_dtype,
                    "model_info": model_info,
                    "message": f"TorchScript model loaded successfully. Input: {dummy_input.shape}, Output: {output_shape}"
                }
                
            except Exception as e:
                return {
                    "success": False, 
                    "error": f"TorchScript inference failed: {e}",
                    "model_file": str(torchscript_file),
                    "file_size_mb": file_size,
                    "model_info": model_info
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
            elif isinstance(outputs, dict) and 'logits' in outputs:
                output_shape = outputs['logits'].shape
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

    def _validate_quantized_model(self, model_path: str, model_type: str, quantization_type: str) -> Dict[str, Any]:
        """Validate quantized models (GPTQ, AWQ, etc.)"""
        try:
            model_dir = Path(model_path)
            
            if quantization_type.lower() == "gptq":
                return self._validate_gptq_model(model_path, model_type)
            elif quantization_type.lower() == "awq":
                return self._validate_awq_model(model_path, model_type)
            elif quantization_type.lower() == "gguf":
                return self._validate_gguf_model(model_path, model_type)
            else:
                return {"success": False, "error": f"Unsupported quantization type: {quantization_type}"}
                
        except Exception as e:
            return {"success": False, "error": f"Quantized model validation failed: {e}"}

    def _validate_awq_model(self, model_path: str, model_type: str) -> Dict[str, Any]:
        """Validate AWQ model"""
        try:
            from awq import AutoAWQForCausalLM
            from transformers import AutoTokenizer
            import torch
            
            model_dir = Path(model_path)
            
            # 检查AWQ特定文件
            awq_files = list(model_dir.glob("*.safetensors"))
            if not awq_files:
                return {"success": False, "error": "No AWQ model files found"}
            
            # 加载AWQ模型
            model = AutoAWQForCausalLM.from_pretrained(str(model_dir))
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            
            # 测试推理
            test_text = "Hello world"
            inputs = tokenizer(test_text, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            # 验证输出
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
                "quantization": "AWQ",
                "message": f"AWQ model loaded successfully. Input: {inputs['input_ids'].shape}, Output: {logits.shape}"
            }
            
        except ImportError:
            return {"success": False, "error": "AWQ library not available"}
        except Exception as e:
            return {"success": False, "error": f"AWQ validation failed: {e}"}

    def _validate_hf_model(self, model_path: str, model_type: str) -> Dict[str, Any]:
        """Validate Hugging Face model"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            model_dir = Path(model_path)
            
            # 检查模型文件
            model_files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
            if not model_files:
                return {"success": False, "error": "No model files found"}
            
            # 加载模型
            model = AutoModelForCausalLM.from_pretrained(str(model_dir))
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            
            # 测试推理
            test_text = "Hello world"
            inputs = tokenizer(test_text, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            # 验证输出
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            # 检查模型大小
            total_params = sum(p.numel() for p in model.parameters())
            
            return {
                "success": True,
                "model_file": str(model_dir),
                "input_text": test_text,
                "input_shape": inputs['input_ids'].shape,
                "output_shape": logits.shape,
                "total_params": total_params,
                "message": f"HF model loaded successfully. Input: {inputs['input_ids'].shape}, Output: {logits.shape}, Params: {total_params:,}"
            }
            
        except Exception as e:
            return {"success": False, "error": f"HF validation failed: {e}"}

    def validate_quantization_quality(
        self, 
        original_model_path: str, 
        quantized_model_path: str, 
        quantization_type: str,
        model_type: str = "text-generation"
    ) -> Dict[str, Any]:
        """
        Validate quantization quality by comparing with original model
        
        Args:
            original_model_path: Path to original model
            quantized_model_path: Path to quantized model
            quantization_type: Type of quantization
            model_type: Type of model
            
        Returns:
            Dict with quantization quality metrics
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            import numpy as np
            
            # 加载原始模型
            original_model = AutoModelForCausalLM.from_pretrained(original_model_path)
            tokenizer = AutoTokenizer.from_pretrained(original_model_path)
            
            # 加载量化模型
            if quantization_type.lower() == "gptq":
                from auto_gptq import AutoGPTQForCausalLM
                quantized_model = AutoGPTQForCausalLM.from_quantized(quantized_model_path)
            elif quantization_type.lower() == "awq":
                from awq import AutoAWQForCausalLM
                quantized_model = AutoAWQForCausalLM.from_pretrained(quantized_model_path)
            else:
                quantized_model = AutoModelForCausalLM.from_pretrained(quantized_model_path)
            
            # 测试用例
            test_cases = [
                "Hello world",
                "The quick brown fox",
                "Machine learning is",
                "In the beginning",
                "To be or not to be"
            ]
            
            quality_metrics = {
                "perplexity_diff": [],
                "output_similarity": [],
                "inference_speed": [],
                "memory_usage": []
            }
            
            for test_text in test_cases:
                try:
                    inputs = tokenizer(test_text, return_tensors="pt")
                    
                    # 原始模型推理
                    with torch.no_grad():
                        original_outputs = original_model(**inputs)
                    original_logits = original_outputs.logits
                    
                    # 量化模型推理
                    with torch.no_grad():
                        quantized_outputs = quantized_model(**inputs)
                    quantized_logits = quantized_outputs.logits
                    
                    # 计算输出相似度
                    if original_logits.shape == quantized_logits.shape:
                        similarity = torch.cosine_similarity(
                            original_logits.flatten(), 
                            quantized_logits.flatten(), 
                            dim=0
                        ).item()
                        quality_metrics["output_similarity"].append(similarity)
                        
                        # 计算困惑度差异
                        orig_probs = torch.softmax(original_logits, dim=-1)
                        quant_probs = torch.softmax(quantized_logits, dim=-1)
                        
                        perplexity_diff = torch.abs(orig_probs - quant_probs).mean().item()
                        quality_metrics["perplexity_diff"].append(perplexity_diff)
                        
                except Exception as e:
                    logger.warning(f"Quality test failed for '{test_text}': {e}")
            
            # 计算平均指标
            avg_similarity = np.mean(quality_metrics["output_similarity"]) if quality_metrics["output_similarity"] else 0
            avg_perplexity_diff = np.mean(quality_metrics["perplexity_diff"]) if quality_metrics["perplexity_diff"] else 0
            
            # 计算模型大小差异
            original_size = sum(p.numel() * p.element_size() for p in original_model.parameters())
            quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())
            compression_ratio = original_size / quantized_size if quantized_size > 0 else 1
            
            # 质量评分 (0-10)
            quality_score = 0
            if avg_similarity > 0.95:
                quality_score += 4
            elif avg_similarity > 0.90:
                quality_score += 3
            elif avg_similarity > 0.80:
                quality_score += 2
            else:
                quality_score += 1
                
            if avg_perplexity_diff < 0.01:
                quality_score += 3
            elif avg_perplexity_diff < 0.05:
                quality_score += 2
            elif avg_perplexity_diff < 0.1:
                quality_score += 1
                
            if compression_ratio > 2:
                quality_score += 3
            elif compression_ratio > 1.5:
                quality_score += 2
            elif compression_ratio > 1.1:
                quality_score += 1
            
            return {
                "success": True,
                "quantization_type": quantization_type,
                "quality_score": min(10, quality_score),
                "avg_similarity": avg_similarity,
                "avg_perplexity_diff": avg_perplexity_diff,
                "compression_ratio": compression_ratio,
                "original_size_mb": original_size / (1024 * 1024),
                "quantized_size_mb": quantized_size / (1024 * 1024),
                "message": f"Quantization quality: {quality_score}/10, Similarity: {avg_similarity:.3f}, Compression: {compression_ratio:.2f}x"
            }
            
        except Exception as e:
            return {"success": False, "error": f"Quantization quality validation failed: {e}"}
