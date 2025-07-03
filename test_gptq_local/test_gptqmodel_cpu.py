import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from gptqmodel import GPTQModel, QuantizeConfig
from transformers import AutoTokenizer
import torch

MODEL_ID = "facebook/opt-125m"
OUTPUT_DIR = "./test_gptq_local/opt-125m-gptqmodel-quantized-cpu"

import shutil
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

print(f"Preparing to quantize: {MODEL_ID}")

quantize_config = QuantizeConfig(
    bits=4,
    group_size=32,
    sym=True,
    desc_act=False,
    dynamic={},
)

# Minimal calibration dataset
calibration_dataset = [
    "Hello, this is a test sentence.",
    "Quantization calibration example.",
    "The quick brown fox jumps over the lazy dog."
]

print("Quantizing with GPTQModel (CPU-only)...")
model = GPTQModel.from_pretrained(MODEL_ID, quantize_config=quantize_config, device="cpu")
model.quantize(calibration_dataset)
model.save_pretrained(OUTPUT_DIR)
print("Quantization complete!")

# Try loading the quantized model and generating text
print("Loading quantized model for inference...")
q_model = GPTQModel.load(OUTPUT_DIR)
result = q_model.generate("Hello, world! This is a test.")[0]
print(q_model.tokenizer.decode(result))
print("Diagnostics complete. Check output above for any successful export/save methods.") 