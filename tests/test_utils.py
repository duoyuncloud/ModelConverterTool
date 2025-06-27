import pytest
from model_converter_tool import utils
import os
from pathlib import Path


def test_safe_filename():
    unsafe = 'a<>:"/\\|?*.txt'
    safe = utils.safe_filename(unsafe)
    assert "<" not in safe and ">" not in safe and ":" not in safe
    assert safe.endswith(".txt")


def test_format_file_size():
    assert utils.format_file_size(0) == "0B"
    assert utils.format_file_size(1024) == "1.0KB"
    assert utils.format_file_size(1024 * 1024) == "1.0MB"


def test_is_valid_model_path(tmp_path):
    # HuggingFace path
    assert utils.is_valid_model_path("hf:bert-base-uncased")
    # Local path
    f = tmp_path / "dummy.txt"
    f.write_text("hi")
    assert utils.is_valid_model_path(str(f))
    # Non-existent path
    assert not utils.is_valid_model_path("/not/exist/path")


def test_get_model_name_from_path():
    assert utils.get_model_name_from_path("hf:bert-base-uncased") == "bert-base-uncased"
    assert utils.get_model_name_from_path("/foo/bar/model") == "model"
