import pytest
from model_converter_tool.converter import ModelConverter
import tempfile
import os
import requests
from unittest.mock import patch, MagicMock


def is_network_available():
    """Check if network is available for downloading models"""
    try:
        requests.get("https://huggingface.co", timeout=5)
        return True
    except:
        return False


@pytest.mark.network
@pytest.mark.skipif(not is_network_available(), reason="Network not available")
def test_convert_fp16_prune():
    converter = ModelConverter()
    result = converter.convert(
        input_source="hf:distilbert-base-uncased",
        output_format="fp16",
        output_path="outputs/test_fp16",
        model_type="text-classification",
        postprocess="prune",
    )
    assert isinstance(result, dict)
    assert "success" in result
    assert "validation" in result or "error" in result


@pytest.mark.network
@pytest.mark.skipif(not is_network_available(), reason="Network not available")
def test_convert_onnx(tmp_path):
    converter = ModelConverter()
    # output_path must be a directory for onnx
    out_dir = tmp_path / "onnx_out"
    out_dir.mkdir()
    result = converter.convert(
        input_source="hf:distilbert-base-uncased",
        output_format="onnx",
        output_path=str(out_dir),
        model_type="text-classification",
        postprocess="simplify",
    )
    assert isinstance(result, dict)
    assert "success" in result
    # postprocess_result should be present if postprocess is set
    assert "postprocess_result" in result


@pytest.mark.network
@pytest.mark.skipif(not is_network_available(), reason="Network not available")
def test_convert_torchscript(tmp_path):
    converter = ModelConverter()
    out_dir = tmp_path / "ts_out"
    out_dir.mkdir()
    result = converter.convert(
        input_source="hf:distilbert-base-uncased",
        output_format="torchscript",
        output_path=str(out_dir),
        model_type="text-classification",
        postprocess="optimize",
    )
    assert isinstance(result, dict)
    assert "success" in result
    assert "postprocess_result" in result


@pytest.mark.network
@pytest.mark.skipif(not is_network_available(), reason="Network not available")
def test_convert_hf(tmp_path):
    converter = ModelConverter()
    out_dir = tmp_path / "hf_out"
    out_dir.mkdir()
    result = converter.convert(
        input_source="hf:distilbert-base-uncased",
        output_format="hf",
        output_path=str(out_dir),
        model_type="text-classification",
    )
    assert isinstance(result, dict)
    assert "success" in result


def test_convert_invalid_format():
    converter = ModelConverter()
    result = converter.convert(
        input_source="hf:distilbert-base-uncased",
        output_format="notarealformat",
        output_path="outputs/invalid",
        model_type="text-classification",
    )
    assert not result["success"]
    assert result["error"] == "input validation failed"


def test_convert_offline_mode():
    converter = ModelConverter()
    result = converter.convert(
        input_source="hf:distilbert-base-uncased",
        output_format="onnx",
        output_path="outputs/offline",
        model_type="text-classification",
        offline_mode=True,
    )
    assert not result["success"]
    assert "offline mode" in result["error"]


def test_convert_input_validation():
    converter = ModelConverter()
    result = converter.convert(
        input_source="",
        output_format="onnx",
        output_path="outputs/empty",
        model_type="text-classification",
    )
    assert not result["success"]
    assert result["error"] == "input validation failed"


@pytest.mark.network
@pytest.mark.skipif(not is_network_available(), reason="Network not available")
def test_batch_convert(tmp_path):
    converter = ModelConverter()
    out1 = tmp_path / "batch1"
    out2 = tmp_path / "batch2"
    tasks = [
        {
            "input_source": "hf:distilbert-base-uncased",
            "output_format": "onnx",
            "output_path": str(out1),
            "model_type": "text-classification",
            "postprocess": "simplify",
        },
        {
            "input_source": "hf:distilbert-base-uncased",
            "output_format": "fp16",
            "output_path": str(out2),
            "model_type": "text-classification",
            "postprocess": "prune",
        },
    ]
    results = converter.batch_convert(tasks, max_workers=1)
    assert isinstance(results, list)
    assert len(results) == 2
    for r in results:
        assert "success" in r
        assert "postprocess_result" in r
    # 至少有一个任务应当成功
    assert any(r["success"] for r in results)


# Mock tests for offline scenarios
@patch("model_converter_tool.converter.ModelConverter.convert")
def test_converter_initialization(mock_convert):
    """Test that converter can be initialized without network access"""
    converter = ModelConverter()
    assert converter is not None


def test_converter_config_loading():
    """Test that converter can load configuration without network access"""
    converter = ModelConverter()
    # This should not require network access
    assert hasattr(converter, "supported_formats")
    assert hasattr(converter, "fast_models")
    assert hasattr(converter, "cache_dir")


def test_converter_methods_exist():
    """Test that all expected methods exist on the converter"""
    converter = ModelConverter()
    assert hasattr(converter, "convert")
    assert hasattr(converter, "batch_convert")
    assert callable(converter.convert)
    assert callable(converter.batch_convert)


def test_converter_validation_methods():
    """Test validation methods work without network access"""
    converter = ModelConverter()
    # Test that validation methods exist and are callable
    assert hasattr(converter, "_validate_conversion_inputs")
    assert callable(converter._validate_conversion_inputs)


@patch("model_converter_tool.converter.ModelConverter.convert")
def test_batch_convert_offline(mock_convert):
    """Test batch convert method works without network access"""
    converter = ModelConverter()
    mock_convert.return_value = {"success": False, "error": "offline mode"}

    tasks = [
        {
            "input_source": "hf:test-model",
            "output_format": "onnx",
            "output_path": "/tmp/test",
            "model_type": "text-classification",
        }
    ]

    results = converter.batch_convert(tasks, max_workers=1)
    assert isinstance(results, list)
    assert len(results) == 1
    assert not results[0]["success"]
