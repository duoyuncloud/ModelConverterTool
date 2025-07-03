from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

# 目标目录
output_dir = Path("gpt2_local_model")
output_dir.mkdir(exist_ok=True)

print(f"正在下载 gpt2 模型到 {output_dir} ...")
model = AutoModelForCausalLM.from_pretrained('gpt2')
model.save_pretrained(output_dir)

print(f"正在下载 gpt2 分词器到 {output_dir} ...")
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.save_pretrained(output_dir)

print(f"下载完成！模型和分词器已保存在 {output_dir} 目录下。") 