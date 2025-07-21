import pytest
from unittest.mock import patch
from model_converter_tool.utils import (
    get_disk_usage,
    format_bytes,
    check_disk_space_safety,
    estimate_model_size,
    check_and_handle_disk_space,
)


def test_format_bytes():
    """Test byte formatting function"""
    assert format_bytes(0) == "0B"
    assert format_bytes(1024) == "1.0KiB"
    assert format_bytes(1024**2) == "1.0MiB"
    assert format_bytes(1024**3) == "1.0GiB"
    assert format_bytes(1024**4) == "1.0TiB"
    assert format_bytes(1500) == "1.5KiB"


def test_get_disk_usage():
    """Test disk usage function"""
    total, used, free = get_disk_usage("/")
    assert total > 0
    assert used >= 0
    assert free >= 0
    assert total >= used + free


def test_check_disk_space_safety():
    """Test disk space safety checking"""
    # Mock disk usage to return known values
    with patch("model_converter_tool.utils.get_disk_usage") as mock_disk_usage:
        # Test case: plenty of space
        mock_disk_usage.return_value = (100 * 1024**3, 50 * 1024**3, 50 * 1024**3)  # 100GB total, 50GB free
        has_enough, info = check_disk_space_safety(1 * 1024**3, 5.0)  # 1GB required, 5GB safety margin
        assert has_enough is True
        assert info["has_enough_for_operation"] is True
        assert info["has_safety_margin"] is True

        # Test case: enough for operation but not enough safety margin
        mock_disk_usage.return_value = (100 * 1024**3, 95 * 1024**3, 5 * 1024**3)  # 5GB free
        has_enough, info = check_disk_space_safety(1 * 1024**3, 5.0)  # 1GB required, 5GB safety margin
        assert has_enough is False
        assert info["has_enough_for_operation"] is True
        assert info["has_safety_margin"] is False

        # Test case: not enough space even for operation
        mock_disk_usage.return_value = (100 * 1024**3, 95 * 1024**3, 5 * 1024**3)  # 5GB free
        has_enough, info = check_disk_space_safety(10 * 1024**3, 5.0)  # 10GB required, 5GB safety margin
        assert has_enough is False
        assert info["has_enough_for_operation"] is False
        assert info["has_safety_margin"] is False


def test_estimate_model_size():
    """Test model size estimation"""
    # Test with known model names
    assert estimate_model_size("bert-base-uncased", "onnx") > 0
    assert estimate_model_size("gpt2", "gguf") > 0
    assert estimate_model_size("llama-2-7b", "gguf") > 0

    # Test quantization effects
    size_without_quant = estimate_model_size("gpt2", "gguf")
    size_with_quant = estimate_model_size("gpt2", "gguf", "q4_k_m")
    assert size_with_quant < size_without_quant

    # Test format effects
    size_onnx = estimate_model_size("gpt2", "onnx")
    size_fp16 = estimate_model_size("gpt2", "safetensors", "fp16")
    assert size_fp16 < size_onnx


@patch("model_converter_tool.utils.check_disk_space_safety")
@patch("model_converter_tool.utils.estimate_model_size")
def test_check_and_handle_disk_space(mock_estimate, mock_check):
    """Test the main disk space checking function"""
    # Mock successful case
    mock_estimate.return_value = 1 * 1024**3  # 1GB
    mock_check.return_value = (
        True,
        {"formatted": {"free": "50.0GiB", "required": "1.0GiB", "safety_margin": "5.0GiB"}},
    )

    result = check_and_handle_disk_space("gpt2", "onnx")
    assert result is True

    # Mock insufficient space case
    mock_check.return_value = (
        False,
        {
            "has_enough_for_operation": False,
            "free_bytes": 0.5 * 1024**3,
            "formatted": {"free": "0.5GiB", "required": "1.0GiB"},
        },
    )

    result = check_and_handle_disk_space("gpt2", "onnx")
    assert result is False


def test_estimate_model_size_with_local_file(tmp_path):
    """Test model size estimation with local file"""
    # Create a dummy model file
    model_file = tmp_path / "test_model.bin"
    model_file.write_bytes(b"x" * (100 * 1024**2))  # 100MB file

    size = estimate_model_size(str(model_file), "onnx")
    assert size > 100 * 1024**2  # Should be larger due to conversion buffer


if __name__ == "__main__":
    pytest.main([__file__])
