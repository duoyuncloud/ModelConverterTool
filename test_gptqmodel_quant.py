from gptqmodel import GPTQModel, QuantizeConfig

model_id = "opt-125m-local"
quant_path = "opt125m_gptqmodel_quantized"

quant_config = QuantizeConfig(
    bits=4,
    group_size=128,
    sym=True,
    desc_act=False,
    damp_percent=0.1,
    static_groups=False,
    true_sequential=True
)

# Longer calibration dataset
calibration_dataset = [
    "This is a much longer calibration sentence that should have more than ten tokens for the quantization process.",
    "Another example of a calibration sentence that is sufficiently long to pass the minimum length requirement for GPTQ quantization.",
    "Quantization calibration requires sentences that are not too short, so this one is also long enough to be valid for the test."
]

print("Calibration dataset:", calibration_dataset)
print("Length:", len(calibration_dataset))

# Quantize
model = GPTQModel.from_pretrained(model_id, quant_config, device="cpu")
print("Loaded model:", model)
print("Model inner model:", getattr(model, 'model', None))

# Debug: print tokenizer and tokenized calibration dataset
try:
    tokenizer = model.tokenizer
    print("Model tokenizer:", tokenizer)
    tokenized = [tokenizer(text, return_tensors="pt") for text in calibration_dataset]
    print("Tokenized calibration dataset:", tokenized)
    print("Tokenized dataset length:", len(tokenized))
except Exception as e:
    print("Tokenizer or tokenization failed:", e)

model.quantize(calibration_dataset)
model.save_pretrained(quant_path)
print("量化完成！") 