from datasets import load_dataset, Dataset

# 3. Load a dataset to finetune on
from datasets import load_dataset
import time
from sentence_transformers import SentenceTransformer
model_id = [
    'thang1943/multilingual-e5-large-v2',
    'thang1943/bge-m3-finetuned',
    'thang1943/vietnamese-bi-encoder-v2',
    'thang1943/vietnamese-sbert-v2',
    'thang1943/bkcare-embed-v2',
]
dataset = load_dataset("meandyou200175/data_split_csv", split="train")
dataset = dataset.rename_column("pos", "positive")
dataset = dataset.rename_column("neg", "negative")

def flatten_columns(example):
    example['pos'] = example['pos'][0] if isinstance(example['pos'], list) else example['pos']
    example['neg'] = example['neg'][0] if isinstance(example['neg'], list) else example['neg']
    return example
for model in model_id:
    model = SentenceTransformer(model, device='cuda')
    sentences = dataset["query"][:10000]
    start = time.time()
    embeddings = model.encode(sentences)
    end = time.time()
    print(end-start)