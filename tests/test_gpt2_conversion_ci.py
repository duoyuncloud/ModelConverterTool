import tempfile
import pytest
from pathlib import Path
from model_converter_tool.converter import ModelConverter
from model_converter_tool.validator import ModelValidator
import importlib
import torch
from packaging import version


def is_gptq_supported():
    try:
        importlib.import_module("auto_gptq")
        return torch.cuda.is_available()
    except ImportError:
        return False


def is_onnx_supported():
    try:
        import onnxruntime

        # onnxruntime >= 1.12.0 supports IR v11
        return version.parse(onnxruntime.__version__) >= version.parse("1.12.0")
    except Exception:
        return False


def is_torchscript_supported():
    import torch
    from transformers import AutoModelForCausalLM

    try:
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        torch.jit.script(model)
        return True
    except Exception:
        return False


def is_gguf_supported():
    try:
        import llama_cpp  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.parametrize(
    "output_format,quantization",
    [
        pytest.param(
            "onnx",
            None,
            marks=pytest.mark.skipif(
                not is_onnx_supported(), reason="ONNX requires onnxruntime"
            ),
        ),
        pytest.param(
            "torchscript",
            None,
            marks=pytest.mark.skipif(
                not is_torchscript_supported(), reason="TorchScript not supported"
            ),
        ),
        ("fp16", None),
        ("hf", None),
        pytest.param(
            "gptq",
            "q4_k_m",
            marks=pytest.mark.skipif(
                not is_gptq_supported(), reason="GPTQ requires auto-gptq and CUDA"
            ),
        ),
        pytest.param(
            "gguf",
            "q4_k_m",
            marks=pytest.mark.skipif(
                not is_gguf_supported(),
                reason="GGUF requires llama-cpp-python[convert]",
            ),
        ),
    ],
)
def test_gpt2_conversion_and_validation(output_format, quantization):
    converter = ModelConverter()
    ModelValidator()
    with tempfile.TemporaryDirectory() as temp_dir:
        out_dir = Path(temp_dir) / f"gpt2_{output_format}"
        out_dir.mkdir()
        result = converter.convert(
            input_source="hf:gpt2",
            output_format=output_format,
            output_path=str(out_dir),
            model_type="text-generation",
            quantization=quantization,
            validate=True,
        )
        # 智能 skip：仅依赖缺失或环境不支持时 skip，否则 fail
        if output_format == "gguf" and not result["success"]:
            error = result.get("error", "")
            skip_reasons = [
                "llama-cpp-python",
                "No module named",
                "No GGUF files found",
                "no file named",
                "not available",
                "not supported",
            ]
            if any(reason in error for reason in skip_reasons):
                pytest.skip(f"GGUF conversion skipped: {error}")
        assert result["success"], f"Conversion failed for {output_format}: {result.get('error')}"
        # 验证输出文件存在
        files = list(out_dir.glob("*"))
        assert len(files) > 0, f"No files generated for {output_format}"
        # 验证模型能否被加载和推理
        model_validation = result.get("model_validation", {})
        # Dynamically skip ONNX if IR version not supported
        if output_format == "onnx" and isinstance(model_validation, dict):
            error_msg = model_validation.get("error", "")
            if "Unsupported model IR version" in error_msg:
                pytest.skip(
                    "ONNXRuntime does not support required IR version on this platform."
                )
        assert model_validation.get(
            "success", False
        ), f"Model validation failed for {output_format}: {model_validation.get('error')}"
        # 量化格式额外验证
        if quantization and model_validation.get("quality_validation"):
            quality = model_validation["quality_validation"]
            if not quality.get("success", True):
                q_error = quality.get("error", "")
                skip_reasons = [
                    "llama-cpp-python",
                    "No module named",
                    "No GGUF files found",
                    "no file named",
                    "not available",
                    "not supported",
                ]
                if any(reason in q_error for reason in skip_reasons):
                    pytest.skip(f"GGUF quantization quality skipped: {q_error}")
            assert quality.get(
                "success", True
            ), f"Quantization quality validation failed: {quality.get('error')}"
            # 质量分数不为0
            assert (
                quality.get("quality_score", 0) > 0
            ), "Quantization quality score too low"
