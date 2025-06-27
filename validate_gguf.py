import subprocess
from pathlib import Path

# llama.cpp的main可执行文件路径（请根据实际情况修改）
LLAMA_CPP_MAIN = "./main"  # 或绝对路径，如 "/Users/duoyun/llama.cpp/main"
PROMPT = "Hello, world!"
GGUF_DIR = "outputs"  # 存放gguf文件的目录


def validate_gguf(gguf_path):
    try:
        result = subprocess.run(
            [LLAMA_CPP_MAIN, "-m", str(gguf_path), "-p", PROMPT, "-n", "1"],
            capture_output=True, text=True, timeout=30
        )
        if "error" in result.stderr.lower() or "failed" in result.stderr.lower():
            print(f"❌ {gguf_path}: 加载失败")
            print(result.stderr)
            return False
        if "llama_model_load_internal" in result.stdout or "llama.cpp" in result.stdout:
            print(f"✅ {gguf_path}: 加载成功")
            return True
        print(f"⚠️ {gguf_path}: 未检测到典型输出，请人工检查")
        print(result.stdout)
        return False
    except Exception as e:
        print(f"❌ {gguf_path}: 运行异常 {e}")
        return False


def main():
    gguf_files = list(Path(GGUF_DIR).rglob("*.gguf"))
    if not gguf_files:
        print("未找到任何GGUF文件")
        return
    for gguf_file in gguf_files:
        print(f"验证: {gguf_file}")
        validate_gguf(gguf_file)


if __name__ == "__main__":
    main() 