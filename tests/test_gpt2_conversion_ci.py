import os
import tempfile
import pytest
from pathlib import Path
from model_converter_tool.converter import ModelConverter
from model_converter_tool.validator import ModelValidator

@pytest.mark.parametrize("output_format,quantization", [
    ("onnx", None),
    ("torchscript", None),
    ("fp16", None),
    ("hf", None),
    ("gptq", "q4_k_m"),
    ("gguf", "q4_k_m"),
])
def test_gpt2_conversion_and_validation(output_format, quantization):
    converter = ModelConverter()
    validator = ModelValidator()
    with tempfile.TemporaryDirectory() as temp_dir:
        out_dir = Path(temp_dir) / f"gpt2_{output_format}"
        out_dir.mkdir()
        result = converter.convert(
            input_source="hf:gpt2",
            output_format=output_format,
            output_path=str(out_dir),
            model_type="text-generation",
            quantization=quantization,
            validate=True
        )
        assert result["success"], f"Conversion failed for {output_format}: {result.get('error')}"
        # 验证输出文件存在
        files = list(out_dir.glob("*"))
        assert len(files) > 0, f"No files generated for {output_format}"
        # 验证模型能否被加载和推理
        model_validation = result.get("model_validation", {})
        assert model_validation.get("success", False), f"Model validation failed for {output_format}: {model_validation.get('error')}"
        # 量化格式额外验证
        if quantization and model_validation.get("quality_validation"):
            quality = model_validation["quality_validation"]
            assert quality.get("success", True), f"Quantization quality validation failed: {quality.get('error')}"
            # 质量分数不为0
            assert quality.get("quality_score", 0) > 0, "Quantization quality score too low" 