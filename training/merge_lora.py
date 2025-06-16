from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel
import torch

tokenizer = AutoTokenizer.from_pretrained("/mnt/data1tb/LLaMA-Factory/saves/Qwen2.5-7B-Instruct/lora/qwen_lora/checkpoint-3000")

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype = torch.float16,
    device_map = "cuda",
)

model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
model = PeftModel.from_pretrained(model, "/mnt/data1tb/LLaMA-Factory/saves/Qwen2.5-7B-Instruct/lora/qwen_lora/checkpoint-3000")
model = model.merge_and_unload()

model.push_to_hub("thang1943/Qwen2.5-7B-Instruct-final", tokenizer=tokenizer, max_shard_size="5GB")
tokenizer.push_to_hub("thang1943/Qwen2.5-7B-Instruct-final")