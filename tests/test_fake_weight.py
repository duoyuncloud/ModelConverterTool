import os
import json
import shutil
import tempfile
import pytest
from pathlib import Path
from transformers import AutoModel, AutoTokenizer, GPT2Config
from model_converter_tool.api import ModelConverterAPI

@pytest.fixture(scope="module")
def api():
    return ModelConverterAPI()

@pytest.fixture(scope="module")
def output_dir():
    d = Path("test_outputs/fake_weight")
    d.mkdir(parents=True, exist_ok=True)
    return d

@pytest.mark.parametrize("model_id", ["gpt2"])
def test_fake_weight_real_model(api, output_dir, model_id):
    """
    Test --fake-weight on a real HuggingFace model. All weights should be zero, shape unchanged.
    """
    output_path = str(output_dir / f"{model_id.replace('/', '_')}_fake_weight")
    result = api.convert_model(
        model_path=model_id,
        output_format="safetensors",
        output_path=output_path,
        model_type="text-generation",
        device="cpu",
        fake_weight=True,
    )
    assert result.success, f"Fake weight conversion failed: {result.error}"
    assert os.path.exists(output_path)
    model = AutoModel.from_pretrained(output_path, local_files_only=True)
    for _, param in model.named_parameters():
        assert param.shape == param.data.shape
        assert param.data.sum().item() == 0


def test_fake_weight_custom_config(api, output_dir):
    """
    Test --fake-weight with a custom config and custom fake_weight_config (custom shape for some weights).
    """
    tmp_model_dir = tempfile.mkdtemp()
    config_obj = GPT2Config(
        vocab_size=16,
        n_embd=4,
        n_layer=1,
        n_head=2,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=0,
        n_positions=8,
        n_ctx=8,
        n_inner=8,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        attn_pdrop=0.1,
        embd_pdrop=0.1,
        resid_pdrop=0.1,
        summary_first_dropout=0.1,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_attention_mask=False,
    )
    with open(f"{tmp_model_dir}/config.json", "w") as f:
        json.dump(config_obj.to_dict(), f, indent=2)
    vocab = {f"token{i}": i for i in range(16)}
    with open(f"{tmp_model_dir}/vocab.json", "w") as f:
        json.dump(vocab, f, indent=2)
    tokenizer_config = {
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "model_max_length": 8,
        "tokenizer_class": "PreTrainedTokenizerFast"
    }
    with open(f"{tmp_model_dir}/tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f, indent=2)
    # Add minimal merges.txt for GPT2TokenizerFast
    with open(f"{tmp_model_dir}/merges.txt", "w") as f:
        f.write("#version: 0.2\na b\nb c\n")
    # Save a complete tokenizer (including tokenizer.json) to the model dir
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.save_pretrained(tmp_model_dir)
    # Find embedding param name
    model = AutoModel.from_config(config_obj)
    emb_param_name = None
    for name, _ in model.named_parameters():
        if "wte.weight" in name:
            emb_param_name = name
            break
    assert emb_param_name is not None
    custom_shape = [16, 4]  # Must match config n_embd
    fake_weight_shape_dict = {emb_param_name: custom_shape}
    output_path = str(output_dir / "custom_fake_weight")
    # tokenizer = AutoTokenizer.from_pretrained(tmp_model_dir, trust_remote_code=True)
    result = api.convert_model(
        model_path=tmp_model_dir,
        output_format="safetensors",
        output_path=output_path,
        model_type="text-generation",
        device="cpu",
        fake_weight=True,
        fake_weight_shape_dict=fake_weight_shape_dict,
        tokenizer=tokenizer,
    )
    assert result.success, f"Fake weight (custom shape) conversion failed: {result.error}"
    assert os.path.exists(output_path)
    model2 = AutoModel.from_pretrained(output_path, local_files_only=True)
    found = False
    for name, param in model2.named_parameters():
        if name == emb_param_name:
            found = True
            assert list(param.shape) == custom_shape
            assert param.data.sum().item() == 0
    assert found
    shutil.rmtree(tmp_model_dir) 