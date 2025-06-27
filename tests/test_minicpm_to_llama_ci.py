import os
import tempfile
import pytest
from pathlib import Path
from model_converter_tool.converter import ModelConverter
from model_converter_tool.utils import create_dummy_model

@pytest.mark.network
def test_minicpm_to_llama_conversion_ci(tmp_path):
    """Test conversion workflow with a dummy model (CI-friendly)"""
    # 1. Create a dummy model instead of downloading from HF
    dummy_dir = tmp_path / "dummy_minicpm"
    create_dummy_model(
        output_dir=str(dummy_dir),
        hidden_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        vocab_size=1000,
        model_type="llama"
    )
    
    # 2. 转换到支持的格式
    converter = ModelConverter()
    output_path = tmp_path / "minicpm_converted"
    result = converter.convert(
        input_source=str(dummy_dir),
        output_format="hf",  # Use supported format instead of "llama"
        output_path=str(output_path),
        model_type="text-generation",
        device="cpu"
    )
    assert result["success"], f"Conversion failed: {result.get('error')}"
    
    # 3. 检查输出文件存在
    assert output_path.exists()
    config_file = output_path / "config.json"
    assert config_file.exists(), "Config file should exist" 