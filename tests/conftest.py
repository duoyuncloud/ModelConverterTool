import pytest
import requests
from model_converter_tool.api import ModelConverterAPI


@pytest.fixture(scope="module")
def api():
    return ModelConverterAPI()


@pytest.fixture(scope="module")
def output_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("test_outputs")
    return d


def is_hf_model_available(model_id):
    url = f"https://huggingface.co/{model_id}/resolve/main/config.json"
    try:
        r = requests.head(url, timeout=5, allow_redirects=True)
        return r.status_code == 200
    except Exception:
        return False
