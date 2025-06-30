import sys
from model_converter_tool.converter import ModelConverter
converter = ModelConverter()
tiny_model = 'sshleifer/tiny-gpt2'
for fmt in ['gptq', 'awq']:
    print(f'== Testing {fmt} with tiny model ==')
    out_path = f'outputs/tiny_{fmt}'
    result = converter.convert(
        input_source=tiny_model,
        output_format=fmt,
        output_path=out_path,
        model_type='text-generation',
        device='cpu',
        validate=True
    )
    assert result.get('success'), f'{fmt} conversion failed'
    assert result.get('validation', True), f'{fmt} validation failed'
    print(f'✅ {fmt} conversion and validation passed') 