import os
from huggingface_hub import HfApi


def main():
    hf_endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    print(f"HF_ENDPOINT: {hf_endpoint}")

    try:
        api = HfApi(endpoint=hf_endpoint)
        print("Trying to list models on Hugging Face (first 5)...")
        models = api.list_models(limit=5)
        for m in models:
            print(f"- {m.modelId}")
        print("Successfully accessed the Hugging Face API!")
    except Exception as e:
        print("Failed to access the Hugging Face API!")
        print(f"Error message: {e}")
        print("\nPlease check your HF_ENDPOINT setting and network connection.")
        print("To set a custom endpoint, set the HF_ENDPOINT environment variable. For example:")
        print("  export HF_ENDPOINT=https://huggingface.co")


if __name__ == "__main__":
    main()
