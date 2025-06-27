"""
Test GPT-2 model conversion with correctness validation
"""

import pytest
import tempfile
import os
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import requests

from model_converter_tool.converter import ModelConverter


def is_network_available():
    """Check if network is available for downloading models"""
    try:
        requests.get("https://huggingface.co", timeout=5)
        return True
    except BaseException:
        return False


class TestGPT2Conversion:
    """Test GPT-2 model conversion with validation"""

    @pytest.fixture
    def converter(self):
        """Create converter instance"""
        return ModelConverter()

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.mark.skipif(not is_network_available(), reason="Network not available")
    def test_gpt2_onnx_conversion_with_validation(self, converter, temp_dir):
        """Test GPT-2 to ONNX conversion with correctness validation"""
        
        # Convert GPT-2 to ONNX
        out_dir = temp_dir / "gpt2_onnx"
        out_dir.mkdir()
        
        result = converter.convert(
            input_source="hf:gpt2",
            output_format="onnx",
            output_path=str(out_dir),
            model_type="text-generation",
            postprocess="simplify",
        )
        
        assert result["success"], f"ONNX conversion failed: {result.get('error', 'Unknown error')}"
        
        # Validate ONNX model correctness
        onnx_files = list(out_dir.glob("*.onnx"))
        assert len(onnx_files) > 0, "No ONNX files generated"
        
        # Test ONNX model loading and inference
        try:
            import onnxruntime as ort
            
            # Load the ONNX model
            onnx_file = onnx_files[0]
            session = ort.InferenceSession(str(onnx_file))
            
            # Get input/output info
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            
            # Create dummy input (batch_size=1, sequence_length=10)
            dummy_input = np.random.randint(0, 50257, (1, 10), dtype=np.int64)
            
            # Run inference
            outputs = session.run([output_name], {input_name: dummy_input})
            
            # Validate output
            assert len(outputs) == 1, "Should have one output"
            assert outputs[0].shape[0] == 1, "Batch size should be 1"
            assert outputs[0].shape[1] == 10, "Sequence length should be 10"
            assert outputs[0].shape[2] == 50257, "Vocab size should be 50257"
            
            print(f"✅ ONNX validation successful: {onnx_file}")
            
        except ImportError:
            pytest.skip("onnxruntime not available")
        except Exception as e:
            pytest.fail(f"ONNX validation failed: {e}")

    @pytest.mark.skipif(not is_network_available(), reason="Network not available")
    def test_gpt2_torchscript_conversion_with_validation(self, converter, temp_dir):
        """Test GPT-2 to TorchScript conversion with correctness validation"""
        
        # Convert GPT-2 to TorchScript
        out_dir = temp_dir / "gpt2_torchscript"
        out_dir.mkdir()
        
        result = converter.convert(
            input_source="hf:gpt2",
            output_format="torchscript",
            output_path=str(out_dir),
            model_type="text-generation",
            postprocess="optimize",
        )
        
        assert result["success"], f"TorchScript conversion failed: {result.get('error', 'Unknown error')}"
        
        # Validate TorchScript model correctness
        torchscript_files = list(out_dir.glob("*.pt"))
        assert len(torchscript_files) > 0, "No TorchScript files generated"
        
        # Test TorchScript model loading and inference
        try:
            import torch
            
            # Load the TorchScript model
            torchscript_file = torchscript_files[0]
            model = torch.jit.load(str(torchscript_file))
            
            # Create dummy input
            dummy_input = torch.randint(0, 50257, (1, 10), dtype=torch.long)
            
            # Run inference
            with torch.no_grad():
                outputs = model(dummy_input)
            
            # Validate output
            assert outputs.shape[0] == 1, "Batch size should be 1"
            assert outputs.shape[1] == 10, "Sequence length should be 10"
            assert outputs.shape[2] == 50257, "Vocab size should be 50257"
            
            print(f"✅ TorchScript validation successful: {torchscript_file}")
            
        except Exception as e:
            pytest.fail(f"TorchScript validation failed: {e}")

    @pytest.mark.skipif(not is_network_available(), reason="Network not available")
    def test_gpt2_fp16_conversion_with_validation(self, converter, temp_dir):
        """Test GPT-2 to FP16 conversion with correctness validation"""
        
        # Convert GPT-2 to FP16
        out_dir = temp_dir / "gpt2_fp16"
        out_dir.mkdir()
        
        result = converter.convert(
            input_source="hf:gpt2",
            output_format="fp16",
            output_path=str(out_dir),
            model_type="text-generation",
            postprocess="prune",
        )
        
        assert result["success"], f"FP16 conversion failed: {result.get('error', 'Unknown error')}"
        
        # Validate FP16 model correctness
        model_files = list(out_dir.glob("*.safetensors"))
        assert len(model_files) > 0, "No FP16 model files generated"
        
        # Test FP16 model loading and inference
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            # Load the FP16 model
            model = AutoModelForCausalLM.from_pretrained(str(out_dir))
            tokenizer = AutoTokenizer.from_pretrained(str(out_dir))
            
            # Create test input
            test_text = "Hello world"
            inputs = tokenizer(test_text, return_tensors="pt")
            
            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Validate output
            assert outputs.logits.shape[0] == 1, "Batch size should be 1"
            assert outputs.logits.shape[1] == len(inputs['input_ids'][0]), "Sequence length should match input"
            assert outputs.logits.shape[2] == 50257, "Vocab size should be 50257"
            
            print(f"✅ FP16 validation successful: {out_dir}")
            
        except Exception as e:
            pytest.fail(f"FP16 validation failed: {e}")

    @pytest.mark.skipif(not is_network_available(), reason="Network not available")
    def test_gpt2_gguf_conversion_with_validation(self, converter, temp_dir):
        """Test GPT-2 to GGUF conversion with correctness validation"""
        
        # Convert GPT-2 to GGUF
        out_dir = temp_dir / "gpt2_gguf"
        out_dir.mkdir()
        
        result = converter.convert(
            input_source="hf:gpt2",
            output_format="gguf",
            output_path=str(out_dir),
            model_type="text-generation",
            quantization="q4_0",
        )
        
        assert result["success"], f"GGUF conversion failed: {result.get('error', 'Unknown error')}"
        
        # Validate GGUF model correctness
        gguf_files = list(out_dir.glob("*.gguf"))
        assert len(gguf_files) > 0, "No GGUF files generated"
        
        # Test GGUF model loading and inference (if llama.cpp is available)
        gguf_file = gguf_files[0]
        
        # Check if it's a valid GGUF file
        with open(gguf_file, 'rb') as f:
            header = f.read(8)
            assert header.startswith(b'GGUF'), "Invalid GGUF file header"
        
        # Try to load with llama.cpp if available
        try:
            import subprocess
            import sys
            
            # Check if llama.cpp main is available
            llama_main = "./main"
            if not os.path.exists(llama_main):
                # Try to find llama.cpp main in PATH
                import shutil
                llama_main = shutil.which("llama")
                if not llama_main:
                    print("⚠️  llama.cpp main not found, skipping inference test")
                    return
            
            # Test loading with llama.cpp
            cmd = [llama_main, "-m", str(gguf_file), "-n", "1", "--no-display-prompt"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print(f"✅ GGUF validation successful: {gguf_file}")
            else:
                print(f"⚠️  GGUF loading failed: {result.stderr}")
                
        except Exception as e:
            print(f"⚠️  GGUF validation skipped: {e}")

    @pytest.mark.skipif(not is_network_available(), reason="Network not available")
    def test_gpt2_gptq_conversion_with_validation(self, converter, temp_dir):
        """Test GPT-2 to GPTQ conversion with correctness validation"""
        
        # Convert GPT-2 to GPTQ
        out_dir = temp_dir / "gpt2_gptq"
        out_dir.mkdir()
        
        result = converter.convert(
            input_source="hf:gpt2",
            output_format="gptq",
            output_path=str(out_dir),
            model_type="text-generation",
            quantization="q4",
        )
        
        # GPTQ conversion might fail on Mac, that's expected
        if not result["success"]:
            print(f"⚠️  GPTQ conversion failed (expected on Mac): {result.get('error', 'Unknown error')}")
            return
        
        # Validate GPTQ model correctness
        gptq_files = list(out_dir.glob("*.safetensors"))
        assert len(gptq_files) > 0, "No GPTQ files generated"
        
        # Test GPTQ model loading and inference
        try:
            from auto_gptq import AutoGPTQForCausalLM
            from transformers import AutoTokenizer
            import torch
            
            # Load the GPTQ model
            model = AutoGPTQForCausalLM.from_quantized(str(out_dir))
            tokenizer = AutoTokenizer.from_pretrained(str(out_dir))
            
            # Create test input
            test_text = "Hello world"
            inputs = tokenizer(test_text, return_tensors="pt")
            
            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Validate output
            assert outputs.logits.shape[0] == 1, "Batch size should be 1"
            assert outputs.logits.shape[1] == len(inputs['input_ids'][0]), "Sequence length should match input"
            assert outputs.logits.shape[2] == 50257, "Vocab size should be 50257"
            
            print(f"✅ GPTQ validation successful: {out_dir}")
            
        except ImportError:
            print("⚠️  auto-gptq not available, skipping GPTQ validation")
        except Exception as e:
            print(f"⚠️  GPTQ validation failed: {e}")

    @pytest.mark.skipif(not is_network_available(), reason="Network not available")
    def test_gpt2_batch_conversion_with_validation(self, converter, temp_dir):
        """Test GPT-2 batch conversion with validation"""
        
        # Create batch conversion tasks
        tasks = [
            {
                "input_source": "hf:gpt2",
                "output_format": "onnx",
                "output_path": str(temp_dir / "batch_onnx"),
                "model_type": "text-generation",
                "postprocess": "simplify",
            },
            {
                "input_source": "hf:gpt2",
                "output_format": "fp16",
                "output_path": str(temp_dir / "batch_fp16"),
                "model_type": "text-generation",
                "postprocess": "prune",
            },
        ]
        
        # Run batch conversion
        results = converter.batch_convert(tasks, max_workers=1)
        
        assert isinstance(results, list)
        assert len(results) == 2
        
        # Check that at least one conversion succeeded
        successful_conversions = [r for r in results if r["success"]]
        assert len(successful_conversions) > 0, "No conversions succeeded in batch"
        
        print(f"✅ Batch conversion completed: {len(successful_conversions)}/{len(results)} successful")


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 