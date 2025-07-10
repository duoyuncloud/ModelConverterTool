from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"

# Download and cache model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Save to local directory
save_dir = "./models/gpt2"
tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)
print(f"Model saved to {save_dir}")
