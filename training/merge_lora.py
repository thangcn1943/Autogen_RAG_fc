from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel
import torch

tokenizer = AutoTokenizer.from_pretrained("/mnt/data1tb/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/train_2025-05-06-21-17-06")

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype = torch.float16,
    device_map = "cuda",
)

model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
model = PeftModel.from_pretrained(model, "/mnt/data1tb/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/train_2025-05-06-21-17-06")
model = model.merge_and_unload()

model.push_to_hub("thang1943/Llama-3.1-8B-in-ViMed", tokenizer=tokenizer, max_shard_size="5GB")
tokenizer.push_to_hub("thang1943/Llama-3.1-8B-in-ViMed")