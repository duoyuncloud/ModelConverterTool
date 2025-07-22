import pytest
from unittest.mock import patch
from model_converter_tool.disk_space import (
    get_disk_usage,
    format_bytes,
    check_disk_space_safety,
    estimate_model_size,
    check_and_handle_disk_space,
)

def format_gib(gib: float) -> str:
    return f"{gib:.1f}GiB"

def make_mock_space_info(free_gib: float, required_gib: float, margin_gib: float, has_enough: bool,
                         has_margin: bool = True) -> tuple:
    return (
        has_enough,
        {
            "free_bytes": int(free_gib * 1024**3),
            "has_enough_for_operation": has_enough or has_margin,
            "has_safety_margin": has_margin,
            "formatted": {
                "free": format_gib(free_gib),
                "required": format_gib(required_gib),
                "safety_margin": format_gib(margin_gib),
                "remaining_after": format_gib(free_gib - required_gib)
            }
        }
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
    with patch("model_converter_tool.disk_space.get_disk_usage") as mock_disk_usage:
        # Plenty of space
        mock_disk_usage.return_value = (100 * 1024**3, 50 * 1024**3, 50 * 1024**3)
        has_enough, info = check_disk_space_safety(1 * 1024**3, 5.0)
        assert has_enough is True
        assert info["has_enough_for_operation"] is True
        assert info["has_safety_margin"] is True

        # Enough for operation but not for margin
        mock_disk_usage.return_value = (100 * 1024**3, 95 * 1024**3, 5 * 1024**3)
        has_enough, info = check_disk_space_safety(1 * 1024**3, 5.0)
        assert has_enough is False
        assert info["has_enough_for_operation"] is True
        assert info["has_safety_margin"] is False

        # Not enough even for operation
        mock_disk_usage.return_value = (100 * 1024**3, 95 * 1024**3, 5 * 1024**3)
        has_enough, info = check_disk_space_safety(10 * 1024**3, 5.0)
        assert has_enough is False
        assert info["has_enough_for_operation"] is False
        assert info["has_safety_margin"] is False

def test_estimate_model_size():
    """Test model size estimation"""
    assert estimate_model_size("bert-base-uncased", "onnx") > 0
    assert estimate_model_size("gpt2", "gguf") > 0
    assert estimate_model_size("llama-2-7b", "gguf") > 0

    size_without_quant = estimate_model_size("gpt2", "gguf")
    size_with_quant = estimate_model_size("gpt2", "gguf", "q4_k_m")
    assert size_with_quant < size_without_quant

    size_onnx = estimate_model_size("gpt2", "onnx")
    size_fp16 = estimate_model_size("gpt2", "safetensors", "fp16")
    assert size_fp16 < size_onnx

@patch("model_converter_tool.disk_space.estimate_model_size")
@patch("model_converter_tool.disk_space.check_disk_space_safety")
def test_check_and_handle_disk_space(mock_check, mock_estimate):
    """Test the main disk space checking function"""
    mock_estimate.return_value = 1 * 1024**3

    # Case 1: Enough space + safety margin
    mock_check.return_value = make_mock_space_info(50, 1, 5, True)
    assert check_and_handle_disk_space("gpt2", "onnx") is True

    # Case 2: Not enough space at all
    mock_check.return_value = make_mock_space_info(0.5, 1, 5, False, has_margin=False)
    assert check_and_handle_disk_space("gpt2", "onnx") is False

def test_estimate_model_size_with_local_file(tmp_path):
    """Test model size estimation with local file"""
    model_file = tmp_path / "test_model.bin"
    model_file.write_bytes(b"x" * (100 * 1024**2))  # 100MB

    size = estimate_model_size(str(model_file), "onnx")
    assert size > 100 * 1024**2  # buffer added

if __name__ == "__main__":
    pytest.main([__file__])
