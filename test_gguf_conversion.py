from model_converter_tool.converter import ModelConverter
from model_converter_tool.validator import ModelValidator
import os

# 1. 配置模型名和输出目录
hf_model_path = "gpt2"  # 使用 HuggingFace Hub 上的小模型 gpt2
output_dir = "outputs/gguf_test"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 2. 转换
converter = ModelConverter()
result = converter.convert(
    input_source=hf_model_path,
    output_format="gguf",
    output_path=output_dir,
    model_type="text-generation",
    quantization=None,  # 可选: "q4_0"等
    device="cpu",
    validate=True
)
print("转换结果：", result)

# 3. 额外验证（进一步验证GGUF文件）
validator = ModelValidator()
validation = validator.validate_converted_model(
    model_path=output_dir,
    output_format="gguf",
    model_type="text-generation"
)
print("验证结果：", validation) 