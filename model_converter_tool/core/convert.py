from model_converter_tool.api import ModelConverterAPI
from model_converter_tool.core.history import append_history_record
import time


def convert_model(
    input_path: str,
    output_path: str,
    to: str = None,
    quant: str = None,
    model_type: str = "auto",
    device: str = "auto",
    use_large_calibration: bool = False,
    dtype: str = None,
    quantization_config: dict = None,
    fake_weight: bool = False,
    fake_weight_shape_dict: dict = None,
    mup2llama: bool = False,
):
    """
    Main entry point for model conversion. Calls the API for validation and conversion. Records the result in the history.
    """
    api = ModelConverterAPI()
    result = api.convert_model(
        model_path=input_path,
        output_format=to,
        output_path=output_path,
        model_type=model_type,
        device=device,
        quantization=quant,
        use_large_calibration=use_large_calibration,
        dtype=dtype,
        quantization_config=quantization_config,
        fake_weight=fake_weight,
        fake_weight_shape_dict=fake_weight_shape_dict,
        mup2llama=mup2llama,
    )
    record = {
        "model_path": input_path,
        "output_format": to,
        "output_path": output_path,
        "status": "completed" if result and result.success else "failed",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if not result or not result.success:
        record["error"] = getattr(result, "error", "Invalid conversion plan")
    append_history_record(record)
    return result
