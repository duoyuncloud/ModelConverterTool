import pytest
from model_converter_tool.validator import ModelValidator
import tempfile
import os
import shutil

def test_validate_model_hf():
    v = ModelValidator()
    # HuggingFace remote path
    result = v.validate_model('hf:bert-base-uncased')
    assert result.details[0].startswith('Validating HuggingFace model:')
    assert isinstance(result.is_valid, bool)

def test_validate_model_local(tmp_path):
    v = ModelValidator()
    # Create minimal ONNX dir
    onnx_dir = tmp_path / 'onnx_model'
    onnx_dir.mkdir()
    (onnx_dir / 'model.onnx').write_bytes(b'onnx')
    result = v.validate_model(str(onnx_dir), model_type='onnx')
    assert result.is_valid
    assert any('model.onnx' in d for d in result.details)
    # Missing file
    empty_dir = tmp_path / 'empty'
    empty_dir.mkdir()
    result2 = v.validate_model(str(empty_dir), model_type='onnx')
    assert not result2.is_valid
    assert any('Missing' in e for e in result2.errors) 