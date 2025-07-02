import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from gptqmodel import GPTQModel, QuantizeConfig
from transformers import AutoTokenizer
import torch

MODEL_ID = "facebook/opt-125m"
OUTPUT_DIR = "./test_gptq_local/opt-125m-gptqmodel-quantized-cpu"

import shutil
import glob

shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

# Print contents of the local model directory for debugging
print(f"Preparing to quantize: {MODEL_ID}")

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

quantize_config = QuantizeConfig(
    bits=4,
    group_size=32,
    sym=True,
    desc_act=False,
    dynamic={},
)

print("Quantizing with GPTQModel (CPU-only)...")
model = GPTQModel.from_pretrained(MODEL_ID, quantize_config=quantize_config, device="cpu")

# 1. Print all attributes/methods
print("\nModel attributes/methods:", dir(model))

# 2. Try HuggingFace save_pretrained on model and model.model
try:
    print("\nTrying model.save_pretrained...")
    model.save_pretrained(OUTPUT_DIR)
    print("model.save_pretrained succeeded!")
except Exception as e:
    print("model.save_pretrained failed:", e)

try:
    print("\nTrying model.model.save_pretrained...")
    model.model.save_pretrained(OUTPUT_DIR + "_inner")
    print("model.model.save_pretrained succeeded!")
except Exception as e:
    print("model.model.save_pretrained failed:", e)

# 3. Try torch.save
try:
    print("\nTrying torch.save(model)...")
    torch.save(model, OUTPUT_DIR + ".pt")
    print("torch.save(model) succeeded!")
except Exception as e:
    print("torch.save(model) failed:", e)

try:
    print("\nTrying torch.save(model.state_dict())...")
    torch.save(model.state_dict(), OUTPUT_DIR + "_state_dict.pt")
    print("torch.save(model.state_dict()) succeeded!")
except Exception as e:
    print("torch.save(model.state_dict()) failed:", e)

# 4. Check for internal/hidden export/save/dump methods
print("\nChecking for internal/hidden export/save/dump methods:")
for attr in dir(model):
    if "export" in attr or "save" in attr or "dump" in attr:
        print(" -", attr)
        try:
            method = getattr(model, attr)
            if callable(method):
                print(f"   Calling {attr}()...")
                result = method(OUTPUT_DIR + f"_{attr}") if 'save_dir' in method.__code__.co_varnames else method()
                print(f"   {attr}() call succeeded: {result}")
        except Exception as e:
            print(f"   {attr}() call failed: {e}")

print("\nDiagnostics complete. Check output above for any successful export/save methods.")

# Try loading the quantized model and generating text
print("Loading quantized model for inference...")
q_model = GPTQModel.load(OUTPUT_DIR)
result = q_model.generate("Hello, world! This is a test.")[0]
print(q_model.tokenizer.decode(result)) 