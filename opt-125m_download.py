from transformers import AutoModelForCausalLM, AutoTokenizer

# 设置模型名称
model_name = "facebook/opt-125m"

# 设置保存路径
model_path = "/Users/duoyun/Desktop/ModelConverterTool/models/facebook_opt_125m"

# 下载模型和 Tokenizer
print("Downloading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 保存模型到指定路径
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

print(f"Model and tokenizer saved to {model_path}")
