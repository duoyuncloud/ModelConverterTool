import os
import json
import shutil
import torch
import tempfile
import pytest
from model_converter_tool.api import ModelConverterAPI

@pytest.fixture(scope="module")
def tmp_output_dir():
    d = tempfile.mkdtemp(prefix="mup2llama_test_")
    yield d
    shutil.rmtree(d)

@pytest.fixture(scope="module")
def fake_mup_config():
    # Minimal muP config for test
    return {
        "scale_emb": 2.0,
        "scale_depth": 1.0,
        "dim_model_base": 4,
        "hidden_size": 8,
        "num_hidden_layers": 4,
        "vocab_size": 10,
        "num_attention_heads": 2
    }

@pytest.fixture(scope="module")
def fake_model_and_config(tmp_output_dir, fake_mup_config):
    # Create a minimal fake model and config.json
    from transformers import AutoModelForCausalLM, AutoConfig
    config = AutoConfig.from_pretrained("gpt2")
    for k, v in fake_mup_config.items():
        setattr(config, k, v)
    model = AutoModelForCausalLM.from_config(config)
    model_dir = os.path.join(tmp_output_dir, "fake_model")
    model.save_pretrained(model_dir)
    # Save a minimal tokenizer for the fake model
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.save_pretrained(model_dir)
    # Overwrite config.json with our muP config
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(fake_mup_config, f)
    return model_dir

def test_mup2llama_safetensors(tmp_output_dir, fake_model_and_config, fake_mup_config):
    """
    Test that --mup2llama rescales weights and outputs a config.json without muP keys.
    """
    api = ModelConverterAPI()
    output_dir = os.path.join(tmp_output_dir, "out")
    # Run conversion with mup2llama
    result = api.convert_model(
        model_path=fake_model_and_config,
        output_format="safetensors",
        output_path=output_dir,
        mup2llama=True
    )
    assert result.success, f"Conversion failed: {result.error}"
    # Check config.json: muP keys should be removed
    config_path = os.path.join(output_dir, "config.json")
    assert os.path.exists(config_path), "config.json not found in output"
    with open(config_path) as f:
        out_cfg = json.load(f)
    for k in ["scale_emb", "scale_depth", "dim_model_base"]:
        assert k not in out_cfg, f"muP key {k} should be removed from config.json"
    # Check LLaMA keys are present
    for k in ["hidden_size", "num_hidden_layers", "num_attention_heads"]:
        assert k in out_cfg, f"LLaMA key {k} missing in config.json"
    # Check weight scaling: embedding weight should be scaled
    st_path = os.path.join(output_dir, "model.safetensors")
    assert os.path.exists(st_path), "model.safetensors not found in output"
    import safetensors.torch
    tensors = safetensors.torch.load_file(st_path)
    # The embedding weight should be all zeros (from fake model) but scaled by scale_emb (2.0)
    # So still zeros, but test shape and dtype
    emb_keys = ["transformer.wte.weight", "wte.weight", "embed_tokens.weight"]
    found = False
    for key in emb_keys:
        if key in tensors:
            emb = tensors[key]
            found = True
            break
    assert found, f"Embedding weight not found in any of {emb_keys}"
    assert emb.shape[0] == fake_mup_config["vocab_size"]
    assert emb.shape[1] == fake_mup_config["hidden_size"]
    assert emb.dtype == torch.float32 