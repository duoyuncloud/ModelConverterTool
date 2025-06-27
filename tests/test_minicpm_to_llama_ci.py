import os
import torch
from model_converter_tool.converter import ModelConverter

def test_minicpm_to_llama_conversion_ci(tmp_path):
    # 1. 转换
    converter = ModelConverter()
    output_path = tmp_path / "minicpm_llama.bin"
    result = converter.convert(
        input_source="katuni4ka/tiny-random-minicpm",
        output_format="llama",
        output_path=str(output_path),
        device="cpu"
    )
    assert result["success"]
    assert os.path.exists(output_path)

    # 2. 检查能否被transformers加载
    from transformers import AutoModelForCausalLM, AutoConfig
    config = AutoConfig.from_pretrained("katuni4ka/tiny-random-minicpm", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=None,
        config=config,
        state_dict=torch.load(output_path, map_location="cpu"),
        trust_remote_code=True
    )
    assert model is not None 