import os
import json
import time
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForMaskedLM

logger = logging.getLogger(__name__)

MODEL_TYPE_MAP = {
    'causal_lm': AutoModelForCausalLM,
    'seq2seq': AutoModelForSeq2SeqLM,
    'encoder': AutoModelForMaskedLM,  # fallback for BERT-like
}

class ModelConverter:
    def __init__(self, cache_dir: str = "model_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

    def detect_model_type(self, model_name: str) -> str:
        config = AutoConfig.from_pretrained(model_name)
        if hasattr(config, 'is_encoder_decoder') and config.is_encoder_decoder:
            return 'seq2seq'
        if hasattr(config, 'architectures') and config.architectures:
            arch = config.architectures[0].lower()
            if 'causallm' in arch or 'gpt' in arch:
                return 'causal_lm'
            if 'seq2seq' in arch or 't5' in arch:
                return 'seq2seq'
            if 'bert' in arch or 'encoder' in arch or 'maskedlm' in arch:
                return 'encoder'
        # fallback
        return 'encoder'

    def load_model_and_tokenizer(self, model_name: str, model_type: Optional[str] = None):
        print(f"DEBUG: load_model_and_tokenizer called with model_name={model_name}, model_type={model_type}")
        if not model_name:
            raise ValueError("model_name must not be None or empty")
        if not model_type:
            model_type = self.detect_model_type(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_class = MODEL_TYPE_MAP.get(model_type, AutoModel)
        model = model_class.from_pretrained(model_name)
        return model, tokenizer, model_type

    def convert_to_gguf(self, model, output_dir, **kwargs):
        import subprocess
        import tempfile
        script_path = os.environ.get('GGUF_CONVERTER_SCRIPT', './llama.cpp/convert-hf-to-gguf.py')
        if not os.path.exists(script_path):
            return {"success": False, "error": "llama.cpp conversion script not found. Set GGUF_CONVERTER_SCRIPT env var."}
        # Save model to a temp directory in HF format
        temp_dir = tempfile.mkdtemp(prefix="hf_for_gguf_")
        model.save_pretrained(temp_dir)
        # Call the conversion script
        gguf_path = os.path.join(output_dir, "model.gguf")
        cmd = [
            "python3", script_path,
            temp_dir, gguf_path
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            if os.path.exists(gguf_path):
                return {"success": True, "output": gguf_path}
            else:
                return {"success": False, "error": "GGUF file not created."}
        except subprocess.CalledProcessError as e:
            return {"success": False, "error": f"GGUF conversion failed: {e.stderr}"}
        finally:
            shutil.rmtree(temp_dir)

    def convert_to_mlx(self, model, output_dir, **kwargs):
        try:
            import mlx.core as mx
            import numpy as np
            weights = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
            npz_path = os.path.join(output_dir, "model.npz")
            np.savez(npz_path, **weights)
            return {"success": True, "output": npz_path}
        except ImportError:
            return {"success": False, "error": "mlx not installed"}
        except Exception as e:
            return {"success": False, "error": f"MLX conversion failed: {e}"}

    def convert(self, model_name: str, target_format: str, options: Optional[Dict[str, Any]] = None, output_dir: Optional[str] = None) -> Dict[str, Any]:
        options = options or {}
        print(f"DEBUG: convert() called with model_name={model_name}, target_format={target_format}, output_dir={output_dir}, options={options}")
        if not output_dir:
            output_dir = tempfile.mkdtemp(prefix=f"{model_name.replace('/', '_')}_{target_format}_", dir="outputs")
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Starting conversion: {model_name} -> {target_format}")
        try:
            print("DEBUG: Loading model and tokenizer...")
            model, tokenizer, model_type = self.load_model_and_tokenizer(model_name)
            print("DEBUG: Model and tokenizer loaded.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            print(f"ERROR: Failed to load model: {e}")
            return {"success": False, "error": f"Failed to load model: {e}"}
        try:
            print(f"DEBUG: Beginning export for target_format={target_format}")
            if target_format == "hf":
                print("DEBUG: Saving in HuggingFace format...")
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                print("DEBUG: Saved in HuggingFace format.")
                return {"success": True, "output": output_dir}
            elif target_format == "onnx":
                print("DEBUG: Preparing dummy input for ONNX export...")
                dummy_input = torch.randint(0, tokenizer.vocab_size, (1, 8)).to(model.device)
                print("DEBUG: Dummy input prepared. Exporting to ONNX...")
                onnx_path = os.path.join(output_dir, "model.onnx")
                torch.onnx.export(model, dummy_input, onnx_path,
                                  input_names=["input_ids"], output_names=["output"],
                                  opset_version=options.get("opset_version", 13), do_constant_folding=True)
                print("DEBUG: Exported to ONNX.")
                print(f"DEBUG: Checking if ONNX file exists: {onnx_path} -> {os.path.exists(onnx_path)}")
                if os.path.exists(onnx_path):
                    return {"status": "success", "output": output_dir}
                else:
                    return {"status": "failed", "error": "ONNX file not created"}
            elif target_format == "torchscript":
                print("DEBUG: Preparing dummy input for TorchScript export...")
                dummy_input = torch.randint(0, tokenizer.vocab_size, (1, 8)).to(model.device)
                print("DEBUG: Dummy input prepared. Tracing model...")
                traced = torch.jit.trace(model, dummy_input)
                print("DEBUG: Model traced. Saving TorchScript...")
                traced.save(os.path.join(output_dir, "model.pt"))
                print("DEBUG: Exported to TorchScript.")
                return {"success": True, "output": output_dir}
            elif target_format == "fp16":
                print("DEBUG: Converting model to FP16...")
                model = model.half()
                print("DEBUG: Model converted to FP16. Saving...")
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                print("DEBUG: Saved in FP16 format.")
                return {"success": True, "output": output_dir}
            elif target_format == "gptq":
                print("DEBUG: Calling convert_to_gptq...")
                return self.convert_to_gptq(model, output_dir)
            elif target_format == "awq":
                print("DEBUG: Calling convert_to_awq...")
                return self.convert_to_awq(model, output_dir)
            elif target_format == "mlx":
                print("DEBUG: Calling convert_to_mlx...")
                return self.convert_to_mlx(model, output_dir)
            elif target_format == "gguf":
                print("DEBUG: Calling convert_to_gguf...")
                return self.convert_to_gguf(model, output_dir)
            elif target_format == "test":
                print("DEBUG: Writing dummy test file...")
                with open(os.path.join(output_dir, "dummy.json"), "w") as f:
                    json.dump({"model": model_name, "format": target_format}, f)
                print("DEBUG: Saved dummy test file.")
                return {"success": True, "output": output_dir}
            else:
                print(f"ERROR: Unknown target format: {target_format}")
                return {"success": False, "error": f"Unknown target format: {target_format}"}
            print("DEBUG: Export complete.")
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            print(f"ERROR: Conversion failed: {e}")
            return {"success": False, "error": f"Conversion failed: {e}"}

    def convert_to_gptq(self, model, output_dir, **kwargs):
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("GPTQ quantization requires a CUDA-enabled GPU. Please run on a Linux/x86_64 machine with CUDA.")
            from auto_gptq import quantize
        except ImportError:
            raise RuntimeError("auto-gptq is not installed. Please install it with 'pip install auto-gptq'.")
        # ... (rest of your quantization logic)

    def convert_to_awq(self, model, output_dir, **kwargs):
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("AWQ quantization requires a CUDA-enabled GPU. Please run on a Linux/x86_64 machine with CUDA.")
            import awq
        except ImportError:
            raise RuntimeError("awq is not installed. Please install it with 'pip install awq'.")
        # ... (rest of your quantization logic) 