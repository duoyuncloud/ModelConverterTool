import sys
from model_converter_tool.converter import ModelConverter
converter = ModelConverter()
formats = ['hf', 'onnx', 'torchscript', 'fp16', 'gguf', 'mlx']
for fmt in formats:
    print(f'== Testing {fmt} with gpt2 ==')
    out_path = f'outputs/gpt2_{fmt}'
    result = converter.convert(
        input_source='gpt2',
        output_format=fmt,
        output_path=out_path,
        model_type='text-generation',
        device='cpu',
        validate=True
    )
    assert result.get('success'), f'{fmt} conversion failed'
    assert result.get('validation', True), f'{fmt} validation failed'
    print(f'✅ {fmt} conversion and validation passed') 