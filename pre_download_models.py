# This script is for developers or advanced users (e.g., developers, testers, or anyone who needs to batch test, demo, or develop with public HuggingFace models).
# Normal users who only want to convert their own local models do NOT need to run this script.
# For most users, simply use --local-path to convert your own models without pre-downloading public models.

from transformers import AutoModel, AutoTokenizer

MODELS = [
    # Text models
    'distilbert-base-uncased',
    'bert-base-uncased',
    'roberta-base',
    'albert-base-v2',
    'prajjwal1/bert-tiny',
    'sshleifer/tiny-gpt2',
    'distilgpt2',
    'sentence-transformers/all-MiniLM-L6-v2',
    'google/mobilebert-uncased',
    # Vision models
    'microsoft/resnet-18',
    'google/vit-base-patch16-224-in21k',
    'nvidia/mit-b0',
    'facebook/deit-tiny-patch16-224',
    # Audio models
    'facebook/wav2vec2-base-960h',
    'superb/hubert-tiny',
]

def download_all():
    for model_name in MODELS:
        print(f'>>> Downloading: {model_name}')
        try:
            AutoModel.from_pretrained(model_name)
        except Exception as e:
            print(f'  [Model] Failed: {e}')
        try:
            AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            print(f'  [Tokenizer] Failed: {e}')

if __name__ == '__main__':
    download_all() 