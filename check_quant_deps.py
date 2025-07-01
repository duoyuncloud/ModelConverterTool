import importlib

quant_deps = [
    "torch",
    "transformers",
    "tokenizers",
    "datasets",
    "gptqmodel",
    "logbar",
    "tokenicer",
    "device_smi",
    "threadpoolctl",
    "PIL",         # Pillow
    "Pillow",
    "safetensors",
    "onnx",
    "onnxruntime",
    "mlx",
    "optimum",
]

missing = []
for dep in quant_deps:
    try:
        importlib.import_module(dep)
    except ImportError:
        missing.append(dep)

if missing:
    print("❌ 缺失依赖：")
    for dep in missing:
        print("  -", dep)
    print("\n请将上述依赖补充到 requirements.txt 并 pip install")
else:
    print("✅ 所有常见量化依赖已安装！") 