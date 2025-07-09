from model_converter_tool.api import ModelConverterAPI

def convert_model(input_path: str, output_path: str, to: str = None, quant: str = None, model_type: str = "auto", device: str = "auto", use_large_calibration: bool = False):
    api = ModelConverterAPI()
    result = api.converter.convert(
        model_name=input_path,
        output_format=to,
        output_path=output_path,
        model_type=model_type,
        device=device,
        quantization=quant,
        use_large_calibration=use_large_calibration
    )
    return result 