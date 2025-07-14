from model_converter_tool.api import ModelConverterAPI
from model_converter_tool.core.history import append_history_record
import time

def convert_model(input_path: str, output_path: str, to: str = None, quant: str = None, model_type: str = "auto", device: str = "auto", use_large_calibration: bool = False, dtype: str = None, quantization_config: dict = None):
    api = ModelConverterAPI()
    result = api.converter.convert(
        model_name=input_path,
        output_format=to,
        output_path=output_path,
        model_type=model_type,
        device=device,
        quantization=quant,
        use_large_calibration=use_large_calibration,
        dtype=dtype,
        quantization_config=quantization_config
    )
    # Record history
    record = {
        "model_path": input_path,
        "output_format": to,
        "output_path": output_path,
        "status": "completed" if result.success else "failed",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if not result.success:
        record["error"] = result.error
    append_history_record(record)
    return result 