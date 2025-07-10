from transformers import AutoModelForCausalLM, AutoTokenizer

# Set model name
model_name = "facebook/opt-125m"

# Set save path
model_path = "/Users/duoyun/Desktop/ModelConverterTool/models/facebook_opt_125m"


# Download model and Tokenizer
print("Downloading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save model to specified path
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

print(f"Model and tokenizer saved to {model_path}")
