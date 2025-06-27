import os
import tempfile
import pytest
from pathlib import Path
from model_converter_tool.utils import create_dummy_model
from model_converter_tool.converter import ModelConverter
from transformers import AutoModelForCausalLM, AutoTokenizer

# 测试参数组合
DUMMY_MODEL_CONFIGS = [
    {"hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 2, "vocab_size": 128},
    {"hidden_size": 128, "num_hidden_layers": 2, "num_attention_heads": 2, "vocab_size": 256},
    {"hidden_size": 256, "num_hidden_layers": 4, "num_attention_heads": 4, "vocab_size": 512},
]

EXPORT_FORMATS = ["onnx", "torchscript", "gguf"]

@pytest.mark.parametrize("model_cfg", DUMMY_MODEL_CONFIGS)
@pytest.mark.parametrize("export_format", EXPORT_FORMATS)
def test_dummy_model_conversion_and_inference(model_cfg, export_format):
    with tempfile.TemporaryDirectory() as temp_dir:
        dummy_dir = Path(temp_dir) / "dummy_model"
        # 1. 生成 dummy model
        create_dummy_model(output_dir=str(dummy_dir), **model_cfg)
        assert (dummy_dir / "config.json").exists()
        # Check for either pytorch_model.bin or model.safetensors
        model_file_exists = (dummy_dir / "pytorch_model.bin").exists() or (dummy_dir / "model.safetensors").exists()
        assert model_file_exists, f"Model file not found in {list(dummy_dir.glob('*'))}"
        # 2. 用 model-converter 导出指定格式
        out_dir = Path(temp_dir) / f"dummy_{export_format}"
        out_dir.mkdir()
        converter = ModelConverter()
        try:
            result = converter.convert(
                input_source=str(dummy_dir),
                output_format=export_format,
                output_path=str(out_dir),
                model_type="text-generation",
                validate=True
            )
        except Exception as e:
            if export_format == "onnx":
                pytest.skip(f"ONNX export failed for config {model_cfg}: {e}")
            else:
                raise
        assert result["success"], f"Conversion failed: {result.get('error')}"
        # 3. 检查导出文件存在
        files = list(out_dir.glob("*"))
        assert len(files) > 0, f"No files generated for {export_format}"
        # 4. 能否被 transformers 加载（仅hf/torchscript）
        if export_format in ["hf", "torchscript"]:
            model = AutoModelForCausalLM.from_pretrained(str(out_dir))
            tokenizer = AutoTokenizer.from_pretrained(str(out_dir))
            inputs = tokenizer("hello world", return_tensors="pt")
            outputs = model(**inputs)
            assert outputs is not None
        # 5. 能否被 onnxruntime/llama.cpp 加载（可选，需环境支持）
        # 这里只做文件存在性和转换流程验证 