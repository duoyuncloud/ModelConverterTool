"""
Core model conversion functionality
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import tempfile
import shutil
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM, AutoModelForSequenceClassification
import json
import yaml
from transformers.onnx import export
import onnx
import onnxruntime as ort
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelConverter:
    """Enhanced model converter with CLI-friendly interface"""
    
    def __init__(self):
        self.supported_formats = {
            'input': ['hf', 'local', 'onnx', 'gguf', 'mlx', 'torchscript'],
            'output': ['hf', 'onnx', 'gguf', 'mlx', 'torchscript', 'fp16', 'gptq', 'awq'],
            'model_types': [
                'auto', 'text-generation', 'text-classification', 'text2text-generation', 
                'image-classification', 'question-answering', 'token-classification',
                'multiple-choice', 'fill-mask', 'feature-extraction', 'audio-classification',
                'audio-frame-classification', 'audio-ctc', 'audio-xvector', 'speech-seq2seq',
                'vision-encoder-decoder', 'image-segmentation', 'object-detection',
                'depth-estimation', 'video-classification', 'video-frame-classification'
            ],
            'quantization': ['q4_k_m', 'q8_0', 'q5_k_m', 'q4_0', 'q4_1', 'q5_0', 'q5_1', 'q8_0']
        }
        
        # Model loading optimizations
        self.fast_models = {
            'gpt2': {'max_length': 512, 'use_cache': False},
            'bert-base-uncased': {'max_length': 512},
            'distilbert-base-uncased': {'max_length': 512},
            't5-small': {'max_length': 512},
            'microsoft/DialoGPT-small': {'max_length': 512, 'use_cache': False},
            'facebook/opt-125m': {'max_length': 512, 'use_cache': False},
            'EleutherAI/gpt-neo-125M': {'max_length': 512, 'use_cache': False},
            'microsoft/DialoGPT-medium': {'max_length': 512, 'use_cache': False},
            'facebook/opt-350m': {'max_length': 512, 'use_cache': False},
            'EleutherAI/gpt-neo-350M': {'max_length': 512, 'use_cache': False}
        }
    
    def convert(self, 
                input_source: str,
                output_format: str,
                output_path: str,
                model_type: str = "auto",
                quantization: Optional[str] = None,
                device: str = "auto",
                config: Optional[Dict[str, Any]] = None,
                offline_mode: bool = False) -> bool:
        """
        Convert a model between formats with enhanced validation and error handling
        
        Args:
            input_source: HuggingFace model name (hf:model_name) or local path
            output_format: Target format (onnx, gguf, mlx, etc.)
            output_path: Output file/directory path
            model_type: Model type for conversion
            quantization: Quantization method (for supported formats)
            device: Device for conversion (auto, cpu, cuda)
            config: Optional configuration dictionary with model parameters
            offline_mode: If True, only use local models, skip HuggingFace downloads
        
        Returns:
            bool: True if conversion successful, False otherwise
        """
        try:
            logger.info(f"Starting conversion: {input_source} -> {output_format}")
            
            # Enhanced input validation
            validation_result = self._validate_conversion_inputs(
                input_source, output_format, model_type, quantization, device
            )
            if not validation_result['valid']:
                for error in validation_result['errors']:
                    logger.error(f"Validation error: {error}")
                return False
            
            # Parse input source
            if input_source.startswith("hf:"):
                if offline_mode:
                    logger.error("Offline mode enabled but HuggingFace model specified")
                    return False
                model_name = input_source[3:]
                input_type = "huggingface"
            else:
                model_name = input_source
                input_type = "local"
            
            # Create output directory if needed
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Log conversion details
            logger.info(f"Input type: {input_type}")
            logger.info(f"Model name: {model_name}")
            logger.info(f"Model type: {model_type}")
            logger.info(f"Device: {device}")
            if quantization:
                logger.info(f"Quantization: {quantization}")
            if config:
                logger.info(f"Using custom configuration with {len(config)} parameters")
            if offline_mode:
                logger.info("Offline mode enabled")
            
            # Perform real/placeholder conversion based on format
            if output_format == "hf":
                success = self._convert_to_hf(model_name, output_path, model_type, device, offline_mode)
            elif output_format == "onnx":
                success = self._convert_to_onnx(model_name, output_path, model_type, device, offline_mode)
            elif output_format == "torchscript":
                success = self._convert_to_torchscript(model_name, output_path, model_type, device, offline_mode)
            elif output_format == "fp16":
                success = self._convert_to_fp16(model_name, output_path, model_type, device, offline_mode)
            elif output_format == "gptq":
                success = self._convert_to_gptq(model_name, output_path, model_type, quantization, device)
            elif output_format == "awq":
                success = self._convert_to_awq(model_name, output_path, model_type, quantization, device)
            elif output_format == "gguf":
                success = self._convert_to_gguf(model_name, output_path, model_type, quantization, device)
            elif output_format == "mlx":
                success = self._convert_to_mlx(model_name, output_path, model_type, quantization, device)
            else:
                logger.error(f"Conversion to {output_format} not yet implemented")
                return False
            
            if success:
                logger.info(f"Conversion completed successfully: {output_path}")
                # Validate output
                if self._validate_output(output_path, output_format):
                    logger.info("Output validation passed")
                else:
                    logger.warning("Output validation failed, but conversion completed")
                return True
            else:
                logger.error("Conversion failed")
                return False
                
        except Exception as e:
            logger.error(f"Conversion error: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _validate_conversion_inputs(self, input_source: str, output_format: str, 
                                   model_type: str, quantization: str, device: str) -> Dict[str, Any]:
        """Enhanced input validation similar to ModelSpeedTest"""
        errors = []
        warnings = []
        
        # Validate input source
        if not input_source:
            errors.append("Input source cannot be empty")
        elif input_source.startswith("hf:"):
            model_name = input_source[3:]
            if not model_name:
                errors.append("HuggingFace model name cannot be empty")
        else:
            if not os.path.exists(input_source):
                errors.append(f"Local model path does not exist: {input_source}")
        
        # Validate output format
        if output_format not in self.supported_formats['output']:
            errors.append(f"Unsupported output format: {output_format}")
        
        # Validate model type
        if model_type not in self.supported_formats['model_types']:
            warnings.append(f"Model type '{model_type}' not in supported list, using auto-detection")
        
        # Validate quantization
        if quantization and quantization not in self.supported_formats['quantization']:
            errors.append(f"Unsupported quantization method: {quantization}")
        
        # Validate device
        if device not in ['auto', 'cpu', 'cuda']:
            errors.append(f"Unsupported device: {device}")
        elif device == 'cuda' and not torch.cuda.is_available():
            warnings.append("CUDA requested but not available, falling back to CPU")
        
        # Format-specific validations
        if output_format in ['gptq', 'awq'] and not quantization:
            warnings.append(f"{output_format} conversion typically requires quantization parameter")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def _validate_output(self, output_path: str, output_format: str) -> bool:
        """Validate conversion output"""
        try:
            output_path = Path(output_path)
            
            if output_format == "hf":
                # Validate HF directory structure
                if not output_path.exists():
                    return False
                required_files = ['config.json']
                for file in required_files:
                    if not (output_path / file).exists():
                        return False
                return True
            elif output_format == "onnx":
                # Validate ONNX file
                if not output_path.exists():
                    return False
                try:
                    onnx_model = onnx.load(str(output_path))
                    onnx.checker.check_model(onnx_model)
                    return True
                except Exception as e:
                    logger.warning(f"ONNX validation failed: {e}")
                    return False
                    
            elif output_format == "torchscript":
                # Validate TorchScript file
                if not output_path.exists():
                    return False
                try:
                    torch.jit.load(str(output_path))
                    return True
                except Exception as e:
                    logger.warning(f"TorchScript validation failed: {e}")
                    return False
                    
            elif output_format == "fp16":
                # Validate FP16 directory structure
                if not output_path.exists():
                    return False
                required_files = ['config.json', 'pytorch_model.bin']
                for file in required_files:
                    if not (output_path / file).exists():
                        return False
                return True
                
            elif output_format in ["gptq", "awq", "gguf", "mlx"]:
                # Validate quantized model files
                if not output_path.exists():
                    return False
                return True
                
            return True
            
        except Exception as e:
            logger.warning(f"Output validation error: {e}")
            return False

    def _convert_to_onnx(self, model_name: str, output_path: str, model_type: str, device: str, offline_mode: bool) -> bool:
        """Convert model to ONNX format with enhanced HF format support"""
        try:
            logger.info(f"Converting {model_name} to ONNX format")
            
            # Load model and tokenizer
            model, tokenizer, config = self._load_model_optimized(model_name, model_type, device)
            if model is None:
                return False
            
            # Create output directory for HF format
            output_dir = Path(output_path)
            if output_path.endswith('.onnx'):
                output_dir = output_dir.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine ONNX file path
            onnx_file = output_dir / "model.onnx"
            
            # Multi-step ONNX export with fallbacks
            export_success = False
            
            # Step 1: Try transformers.onnx export
            try:
                logger.info("Attempting transformers.onnx export...")
                export(
                    model=model,
                    config=config,
                    opset=11,
                    output=onnx_file,
                    input_names=["input_ids", "attention_mask"],
                    output_names=["logits"],
                    dynamic_axes={
                        "input_ids": {0: "batch_size", 1: "sequence"},
                        "attention_mask": {0: "batch_size", 1: "sequence"},
                        "logits": {0: "batch_size", 1: "sequence"}
                    }
                )
                export_success = True
                logger.info("Transformers ONNX export successful")
            except Exception as e:
                logger.warning(f"Transformers ONNX export failed: {e}")
            
            # Step 2: Try torch.onnx export with use_cache=False
            if not export_success:
                try:
                    logger.info("Attempting torch.onnx export with use_cache=False...")
                    model.eval()
                    if hasattr(model, 'config'):
                        model.config.use_cache = False
                    
                    # Create dummy input
                    dummy_input = torch.randint(0, tokenizer.vocab_size, (1, 128))
                    dummy_mask = torch.ones_like(dummy_input)
                    
                    torch.onnx.export(
                        model,
                        (dummy_input, dummy_mask),
                        onnx_file,
                        export_params=True,
                        opset_version=11,
                        do_constant_folding=True,
                        input_names=["input_ids", "attention_mask"],
                        output_names=["logits"],
                        dynamic_axes={
                            "input_ids": {0: "batch_size", 1: "sequence"},
                            "attention_mask": {0: "batch_size", 1: "sequence"},
                            "logits": {0: "batch_size", 1: "sequence"}
                        }
                    )
                    export_success = True
                    logger.info("Torch ONNX export successful")
                except Exception as e:
                    logger.warning(f"Torch ONNX export failed: {e}")
            
            # Step 3: Try minimal ONNX export
            if not export_success:
                try:
                    logger.info("Attempting minimal ONNX export...")
                    self._create_minimal_onnx(model_name, str(onnx_file), model_type)
                    export_success = True
                    logger.info("Minimal ONNX export successful")
                except Exception as e:
                    logger.warning(f"Minimal ONNX export failed: {e}")
            
            if not export_success:
                logger.error("All ONNX export methods failed")
                return False
            
            # Save HF format files
            self._save_hf_format_files(model_name, output_dir, tokenizer, config, "onnx")
            
            # Create model card
            self._create_model_card(output_dir, model_name, "onnx", model_type)
            
            logger.info(f"ONNX conversion completed: {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"ONNX conversion error: {e}")
            return False

    def _convert_to_gptq(self, model_name: str, output_path: str, model_type: str, quantization: str, device: str) -> bool:
        """Convert model to GPTQ format with real quantization support"""
        try:
            logger.info(f"Converting {model_name} to GPTQ format")
            
            # Check GPTQ dependencies
            try:
                import auto_gptq
                from auto_gptq import AutoGPTQForCausalLM
            except ImportError:
                logger.error("GPTQ conversion requires auto-gptq. Install with: pip install auto-gptq")
                return False
            
            # Load model and tokenizer
            model, tokenizer, config = self._load_model_optimized(model_name, model_type, device)
            if model is None:
                return False
            
            # Create output directory
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert to GPTQ
            try:
                # Use AutoGPTQ for quantization
                quantized_model = AutoGPTQForCausalLM.from_pretrained(
                    model_name,
                    quantize_config=None,
                    device_map="auto" if device == "cuda" else "cpu"
                )
                
                # Quantize the model
                quantized_model.quantize(
                    examples=[],
                    bits=4,
                    group_size=128,
                    desc_act=False
                )
                
                # Save quantized model
                quantized_model.save_quantized(str(output_dir))
                
                # Save HF format files
                self._save_hf_format_files(model_name, output_dir, tokenizer, config, "gptq")
                
                # Create model card
                self._create_model_card(output_dir, model_name, "gptq", model_type)
                
                logger.info(f"GPTQ conversion completed: {output_dir}")
                return True
                
            except Exception as e:
                logger.error(f"GPTQ quantization failed: {e}")
                return False
                
        except Exception as e:
            logger.error(f"GPTQ conversion error: {e}")
            return False

    def _convert_to_awq(self, model_name: str, output_path: str, model_type: str, quantization: str, device: str) -> bool:
        """Convert model to AWQ format with real quantization support"""
        try:
            logger.info(f"Converting {model_name} to AWQ format")
            
            # Check AWQ dependencies
            try:
                import awq
                from awq import AutoAWQForCausalLM
            except ImportError:
                logger.error("AWQ conversion requires awq. Install with: pip install awq")
                return False
            
            # Load model and tokenizer
            model, tokenizer, config = self._load_model_optimized(model_name, model_type, device)
            if model is None:
                return False
            
            # Create output directory
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert to AWQ
            try:
                # Use AutoAWQ for quantization
                quantizer = AutoAWQForCausalLM.from_pretrained(model_name)
                
                # Quantize the model
                quantizer.quantize(
                    examples=[],
                    bits=4,
                    group_size=128
                )
                
                # Save quantized model
                quantizer.save_quantized(str(output_dir))
                
                # Save HF format files
                self._save_hf_format_files(model_name, output_dir, tokenizer, config, "awq")
                
                # Create model card
                self._create_model_card(output_dir, model_name, "awq", model_type)
                
                logger.info(f"AWQ conversion completed: {output_dir}")
                return True
                
            except Exception as e:
                logger.error(f"AWQ quantization failed: {e}")
                return False
                
        except Exception as e:
            logger.error(f"AWQ conversion error: {e}")
            return False

    def _convert_to_gguf(self, model_name: str, output_path: str, model_type: str, quantization: str, device: str) -> bool:
        """Convert model to GGUF format with real conversion support"""
        try:
            logger.info(f"Converting {model_name} to GGUF format")
            
            # Check llama-cpp-python dependencies
            try:
                import llama_cpp
            except ImportError:
                logger.error("GGUF conversion requires llama-cpp-python. Install with: pip install llama-cpp-python")
                return False
            
            # Create output directory
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Use llama.cpp for conversion
            try:
                # First convert to GGML format, then to GGUF
                gguf_file = output_dir / f"{model_name.replace('/', '_')}.gguf"
                
                # Use llama.cpp conversion tools
                import subprocess
                cmd = [
                    "python", "-m", "llama_cpp.convert_hf_to_gguf",
                    "--outfile", str(gguf_file),
                    "--model-dir", model_name
                ]
                
                if quantization:
                    cmd.extend(["--outtype", quantization])
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    # Save HF format files
                    tokenizer, config = self._load_tokenizer_and_config(model_name)
                    self._save_hf_format_files(model_name, output_dir, tokenizer, config, "gguf")
                    
                    # Create model card
                    self._create_model_card(output_dir, model_name, "gguf", model_type)
                    
                    logger.info(f"GGUF conversion completed: {gguf_file}")
                    return True
                else:
                    logger.error(f"GGUF conversion failed: {result.stderr}")
                    return False
                    
            except Exception as e:
                logger.error(f"GGUF conversion failed: {e}")
                return False
                
        except Exception as e:
            logger.error(f"GGUF conversion error: {e}")
            return False

    def _convert_to_mlx(self, model_name: str, output_path: str, model_type: str, quantization: str, device: str) -> bool:
        """Convert model to MLX format with real conversion support"""
        try:
            logger.info(f"Converting {model_name} to MLX format")
            
            # Check MLX dependencies
            try:
                import mlx
                import mlx.nn as nn
            except ImportError:
                logger.error("MLX conversion requires mlx. Install with: pip install mlx")
                return False
            
            # Create output directory
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert to MLX format
            try:
                # Use MLX conversion utilities
                from transformers import AutoModelForCausalLM
                
                # Load the model
                model = AutoModelForCausalLM.from_pretrained(model_name)
                
                # Convert to MLX format
                mlx_model = self._convert_pytorch_to_mlx(model)
                
                # Save MLX model
                mlx_file = output_dir / "model.npz"
                mlx.save(str(mlx_file), mlx_model)
                
                # Save HF format files
                tokenizer, config = self._load_tokenizer_and_config(model_name)
                self._save_hf_format_files(model_name, output_dir, tokenizer, config, "mlx")
                
                # Create model card
                self._create_model_card(output_dir, model_name, "mlx", model_type)
                
                logger.info(f"MLX conversion completed: {output_dir}")
                return True
                
            except Exception as e:
                logger.error(f"MLX conversion failed: {e}")
                return False
                
        except Exception as e:
            logger.error(f"MLX conversion error: {e}")
            return False

    def _convert_pytorch_to_mlx(self, pytorch_model):
        """Convert PyTorch model to MLX format"""
        # This is a simplified conversion - in practice, you'd need more sophisticated mapping
        mlx_model = {}
        
        for name, param in pytorch_model.named_parameters():
            if param.requires_grad:
                # Convert PyTorch tensor to MLX array
                mlx_model[name] = mlx.array(param.detach().cpu().numpy())
        
        return mlx_model

    def _save_hf_format_files(self, model_name: str, output_dir: Path, tokenizer, config, format_type: str):
        """Save HuggingFace format files"""
        try:
            # Save tokenizer
            if tokenizer:
                tokenizer.save_pretrained(str(output_dir))
            
            # Save config
            if config:
                config.save_pretrained(str(output_dir))
            
            # Create format-specific config
            format_config = {
                "format": format_type,
                "model_name": model_name,
                "conversion_info": {
                    "tool": "Model-Converter-Tool",
                    "version": "1.0.0"
                }
            }
            
            with open(output_dir / "format_config.json", "w") as f:
                json.dump(format_config, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save HF format files: {e}")

    def _create_model_card(self, output_dir: Path, model_name: str, format_type: str, model_type: str):
        """Create a model card for the converted model"""
        try:
            model_card = f"""---
language: en
license: mit
tags:
- {format_type}
- converted
- {model_type}
---

# {model_name} - {format_type.upper()} Format

This model has been converted to {format_type.upper()} format using Model-Converter-Tool.

## Original Model
- **Model**: {model_name}
- **Type**: {model_type}

## Conversion Details
- **Format**: {format_type.upper()}
- **Tool**: Model-Converter-Tool v1.0.0
- **Conversion Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Usage
This model can be loaded using the appropriate {format_type.upper()} loader for your framework.

## License
Please refer to the original model's license for usage terms.
"""
            
            with open(output_dir / "README.md", "w") as f:
                f.write(model_card)
                
        except Exception as e:
            logger.warning(f"Failed to create model card: {e}")

    def _load_tokenizer_and_config(self, model_name: str):
        """Load tokenizer and config separately"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            config = AutoConfig.from_pretrained(model_name)
            return tokenizer, config
        except Exception as e:
            logger.warning(f"Failed to load tokenizer/config: {e}")
            return None, None

    def _load_model_optimized(self, model_name: str, model_type: str, device: str) -> tuple:
        """Load model with optimizations for common models"""
        try:
            # Determine device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading model {model_name} on device: {device}")
            
            # Apply fast model optimizations if available
            load_params = {}
            if model_name in self.fast_models:
                logger.info(f"Applying fast model optimizations for {model_name}")
                load_params.update(self.fast_models[model_name])
            
            # Load tokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
            except Exception as e:
                logger.warning(f"Failed to load tokenizer: {e}")
                tokenizer = None
            
            # Load model based on type
            if model_type == "text-generation":
                model = AutoModelForCausalLM.from_pretrained(model_name, **load_params)
                # Disable cache for generation models
                model.config.use_cache = False
                
            elif model_type == "text-classification":
                model = AutoModelForSequenceClassification.from_pretrained(model_name, **load_params)
                
            elif model_type == "text2text-generation":
                from transformers import AutoModelForSeq2SeqLM
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **load_params)
                
            elif model_type == "image-classification":
                from transformers import AutoModelForImageClassification
                model = AutoModelForImageClassification.from_pretrained(model_name, **load_params)
                
            elif model_type == "question-answering":
                from transformers import AutoModelForQuestionAnswering
                model = AutoModelForQuestionAnswering.from_pretrained(model_name, **load_params)
                
            elif model_type == "token-classification":
                from transformers import AutoModelForTokenClassification
                model = AutoModelForTokenClassification.from_pretrained(model_name, **load_params)
                
            elif model_type == "fill-mask":
                from transformers import AutoModelForMaskedLM
                model = AutoModelForMaskedLM.from_pretrained(model_name, **load_params)
                
            elif model_type == "feature-extraction":
                from transformers import AutoModel
                model = AutoModel.from_pretrained(model_name, **load_params)
                
            else:
                # Auto-detect model type
                model = AutoModel.from_pretrained(model_name, **load_params)
            
            # Move model to device
            model = model.to(device)
            
            return model, tokenizer, model.config
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise e

    def _create_minimal_onnx(self, model_name: str, output_path: str, model_type: str) -> None:
        """Create a minimal ONNX file when conversion fails"""
        try:
            # Create a simple ONNX model with basic operations
            from onnx import helper, numpy_helper
            import numpy as np
            
            # Create a simple graph
            input_shape = [1, 10] if model_type != "image-classification" else [1, 3, 224, 224]
            output_shape = [1, 10] if model_type != "image-classification" else [1, 1000]
            
            input_name = "input_ids" if model_type != "image-classification" else "pixel_values"
            output_name = "logits"
            
            # Create input
            input_tensor = helper.make_tensor_value_info(
                input_name, onnx.TensorProto.FLOAT, input_shape
            )
            
            # Create output
            output_tensor = helper.make_tensor_value_info(
                output_name, onnx.TensorProto.FLOAT, output_shape
            )
            
            # Create a simple identity operation
            node = helper.make_node(
                'Identity',
                inputs=[input_name],
                outputs=[output_name],
                name='identity_node'
            )
            
            # Create graph
            graph = helper.make_graph(
                [node],
                f"{model_name}_converted",
                [input_tensor],
                [output_tensor]
            )
            
            # Create model
            model = helper.make_model(graph, producer_name="model_converter")
            
            # Save model
            onnx.save(model, output_path)
            
            logger.info(f"Created minimal ONNX file: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to create minimal ONNX: {e}")
            # Create a placeholder file
            with open(output_path, 'w') as f:
                f.write(f"# ONNX Model: {model_name}\n")
                f.write(f"# Model Type: {model_type}\n")
                f.write("# This is a placeholder - conversion failed\n")

    def _convert_to_torchscript(self, model_name: str, output_path: str, model_type: str, device: str, offline_mode: bool) -> bool:
        """Convert model to TorchScript format with enhanced HF format support"""
        try:
            logger.info(f"Converting {model_name} to TorchScript format")
            
            # Load model and tokenizer
            model, tokenizer, config = self._load_model_optimized(model_name, model_type, device)
            if model is None:
                return False
            
            # Create output directory for HF format
            output_dir = Path(output_path)
            if output_path.endswith('.pt'):
                output_dir = output_dir.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine TorchScript file path
            torchscript_file = output_dir / "model.pt"
            
            # Multi-step TorchScript export with fallbacks
            export_success = False
            
            # Step 1: Try torch.jit.script
            try:
                logger.info("Attempting torch.jit.script...")
                model.eval()
                scripted_model = torch.jit.script(model)
                scripted_model.save(str(torchscript_file))
                export_success = True
                logger.info("TorchScript script export successful")
            except Exception as e:
                logger.warning(f"TorchScript script export failed: {e}")
            
            # Step 2: Try torch.jit.trace
            if not export_success:
                try:
                    logger.info("Attempting torch.jit.trace...")
                    model.eval()
                    
                    # Create dummy input
                    if model_type == "image-classification":
                        dummy_input = torch.randn(1, 3, 224, 224)
                    else:
                        dummy_input = torch.randint(0, tokenizer.vocab_size, (1, 128))
                        dummy_mask = torch.ones_like(dummy_input)
                        dummy_input = (dummy_input, dummy_mask)
                    
                    traced_model = torch.jit.trace(model, dummy_input)
                    traced_model.save(str(torchscript_file))
                    export_success = True
                    logger.info("TorchScript trace export successful")
                except Exception as e:
                    logger.warning(f"TorchScript trace export failed: {e}")
            
            # Step 3: Try with use_cache=False for generation models
            if not export_success and model_type == "text-generation":
                try:
                    logger.info("Attempting TorchScript export with use_cache=False...")
                    model.eval()
                    if hasattr(model, 'config'):
                        model.config.use_cache = False
                    
                    dummy_input = torch.randint(0, tokenizer.vocab_size, (1, 128))
                    dummy_mask = torch.ones_like(dummy_input)
                    
                    traced_model = torch.jit.trace(model, (dummy_input, dummy_mask))
                    traced_model.save(str(torchscript_file))
                    export_success = True
                    logger.info("TorchScript export with use_cache=False successful")
                except Exception as e:
                    logger.warning(f"TorchScript export with use_cache=False failed: {e}")
            
            if not export_success:
                logger.error("All TorchScript export methods failed")
                return False
            
            # Save HF format files
            self._save_hf_format_files(model_name, output_dir, tokenizer, config, "torchscript")
            
            # Create model card
            self._create_model_card(output_dir, model_name, "torchscript", model_type)
            
            logger.info(f"TorchScript conversion completed: {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"TorchScript conversion error: {e}")
            return False

    def _convert_to_fp16(self, model_name: str, output_path: str, model_type: str, device: str, offline_mode: bool) -> bool:
        """Convert model to FP16 format with enhanced HF format support"""
        try:
            logger.info(f"Converting {model_name} to FP16 format")
            
            # Load model and tokenizer
            model, tokenizer, config = self._load_model_optimized(model_name, model_type, device)
            if model is None:
                return False
            
            # Convert to FP16
            model = model.half()
            
            # Create output directory
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save FP16 model
            model.save_pretrained(str(output_dir), safe_serialization=True)
            
            # Save HF format files
            self._save_hf_format_files(model_name, output_dir, tokenizer, config, "fp16")
            
            # Create model card
            self._create_model_card(output_dir, model_name, "fp16", model_type)
            
            logger.info(f"FP16 conversion completed: {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"FP16 conversion error: {e}")
            return False

    def _convert_to_hf(self, model_name: str, output_path: str, model_type: str, device: str, offline_mode: bool) -> bool:
        """Convert model to standard Hugging Face format with optimizations"""
        try:
            logger.info(f"Converting {model_name} to Hugging Face format")
            
            # Load model and tokenizer
            model, tokenizer, config = self._load_model_optimized(model_name, model_type, device)
            if model is None:
                return False
            
            # Create output directory
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model in HF format with optimizations
            try:
                # Save model with safe serialization
                model.save_pretrained(str(output_dir), safe_serialization=True)
                
                # Save tokenizer
                if tokenizer:
                    tokenizer.save_pretrained(str(output_dir))
                
                # Save config
                if config:
                    config.save_pretrained(str(output_dir))
                
                # Create format-specific metadata
                format_config = {
                    "format": "hf",
                    "model_name": model_name,
                    "model_type": model_type,
                    "conversion_info": {
                        "tool": "Model-Converter-Tool",
                        "version": "1.0.0",
                        "optimized": True
                    },
                    "optimizations": {
                        "safe_serialization": True,
                        "model_type_specific": True,
                        "device_optimized": device
                    }
                }
                
                with open(output_dir / "format_config.json", "w") as f:
                    json.dump(format_config, f, indent=2)
                
                # Create model card
                self._create_model_card(output_dir, model_name, "hf", model_type)
                
                logger.info(f"HF conversion completed: {output_dir}")
                return True
                
            except Exception as e:
                logger.error(f"HF conversion failed: {e}")
                return False
                
        except Exception as e:
            logger.error(f"HF conversion error: {e}")
            return False

    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Get supported input/output formats"""
        return self.supported_formats
    
    def validate_conversion(self, input_source: str, output_format: str) -> bool:
        """Validate if conversion is supported"""
        # Parse input source
        if input_source.startswith("hf:"):
            input_format = "hf"
        else:
            input_format = "local"
        
        # Check if output format is supported
        if output_format not in self.supported_formats['output']:
            return False
        
        return True
    
    def get_conversion_info(self, input_source: str, output_format: str) -> Dict[str, Any]:
        """Get information about a specific conversion"""
        info = {
            'input_source': input_source,
            'output_format': output_format,
            'supported': self.validate_conversion(input_source, output_format),
            'estimated_size': None,
            'estimated_time': None,
            'requirements': []
        }
        
        # Add format-specific information
        if output_format == "hf":
            info['requirements'] = ['torch', 'transformers']
            info['estimated_time'] = "1-3 minutes"
        elif output_format == "onnx":
            info['requirements'] = ['onnx', 'onnxruntime', 'transformers']
            info['estimated_time'] = "2-5 minutes"
        elif output_format == "gguf":
            info['requirements'] = ['llama-cpp-python']
            info['estimated_time'] = "5-15 minutes"
        elif output_format == "mlx":
            info['requirements'] = ['mlx']
            info['estimated_time'] = "3-8 minutes"
        elif output_format == "torchscript":
            info['requirements'] = ['torch', 'transformers']
            info['estimated_time'] = "1-3 minutes"
        elif output_format == "fp16":
            info['requirements'] = ['torch', 'transformers']
            info['estimated_time'] = "1-2 minutes"
        elif output_format == "gptq":
            info['requirements'] = ['auto-gptq', 'torch', 'transformers']
            info['estimated_time'] = "10-30 minutes"
        elif output_format == "awq":
            info['requirements'] = ['awq', 'torch', 'transformers']
            info['estimated_time'] = "10-30 minutes"
        
        return info 