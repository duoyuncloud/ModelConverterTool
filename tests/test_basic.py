import pytest
from model_converter_tool.converter import ModelConverter
import tempfile
import os

def test_convert_fp16_prune():
    converter = ModelConverter()
    result = converter.convert(
        input_source="hf:distilbert-base-uncased",
        output_format="fp16",
        output_path="outputs/test_fp16",
        model_type="text-classification",
        postprocess="prune"
    )
    assert isinstance(result, dict)
    assert 'success' in result
    assert 'validation' in result or 'error' in result

def test_convert_onnx(tmp_path):
    converter = ModelConverter()
    # output_path must be a directory for onnx
    out_dir = tmp_path / 'onnx_out'
    out_dir.mkdir()
    result = converter.convert(
        input_source="hf:distilbert-base-uncased",
        output_format="onnx",
        output_path=str(out_dir),
        model_type="text-classification",
        postprocess="simplify"
    )
    assert isinstance(result, dict)
    assert 'success' in result
    # postprocess_result should be present if postprocess is set
    assert 'postprocess_result' in result

def test_convert_torchscript(tmp_path):
    converter = ModelConverter()
    out_dir = tmp_path / 'ts_out'
    out_dir.mkdir()
    result = converter.convert(
        input_source="hf:distilbert-base-uncased",
        output_format="torchscript",
        output_path=str(out_dir),
        model_type="text-classification",
        postprocess="optimize"
    )
    assert isinstance(result, dict)
    assert 'success' in result
    assert 'postprocess_result' in result

def test_convert_hf(tmp_path):
    converter = ModelConverter()
    out_dir = tmp_path / 'hf_out'
    out_dir.mkdir()
    result = converter.convert(
        input_source="hf:distilbert-base-uncased",
        output_format="hf",
        output_path=str(out_dir),
        model_type="text-classification"
    )
    assert isinstance(result, dict)
    assert 'success' in result

def test_convert_invalid_format():
    converter = ModelConverter()
    result = converter.convert(
        input_source="hf:distilbert-base-uncased",
        output_format="notarealformat",
        output_path="outputs/invalid",
        model_type="text-classification"
    )
    assert not result['success']
    assert result['error'] == 'input validation failed'

def test_convert_offline_mode():
    converter = ModelConverter()
    result = converter.convert(
        input_source="hf:distilbert-base-uncased",
        output_format="onnx",
        output_path="outputs/offline",
        model_type="text-classification",
        offline_mode=True
    )
    assert not result['success']
    assert 'offline mode' in result['error']

def test_convert_input_validation():
    converter = ModelConverter()
    result = converter.convert(
        input_source="",
        output_format="onnx",
        output_path="outputs/empty",
        model_type="text-classification"
    )
    assert not result['success']
    assert result['error'] == 'input validation failed'

def test_batch_convert(tmp_path):
    converter = ModelConverter()
    out1 = tmp_path / 'batch1'
    out2 = tmp_path / 'batch2'
    tasks = [
        {
            'input_source': 'hf:distilbert-base-uncased',
            'output_format': 'onnx',
            'output_path': str(out1),
            'model_type': 'text-classification',
            'postprocess': 'simplify'
        },
        {
            'input_source': 'hf:distilbert-base-uncased',
            'output_format': 'fp16',
            'output_path': str(out2),
            'model_type': 'text-classification',
            'postprocess': 'prune'
        }
    ]
    results = converter.batch_convert(tasks, max_workers=1)
    assert isinstance(results, list)
    assert len(results) == 2
    for r in results:
        assert 'success' in r
        assert 'postprocess_result' in r
    # 至少有一个任务应当成功
    assert any(r['success'] for r in results) 