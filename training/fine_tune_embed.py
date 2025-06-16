import torch
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    SequentialEvaluator,
)
from sentence_transformers.util import cos_sim
from datasets import load_dataset, concatenate_datasets
from sentence_transformers import SentenceTransformerModelCardData, SentenceTransformer
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss, OnlineContrastiveLoss, ContrastiveLoss, GISTEmbedLoss, TripletLoss
from sentence_transformers import SentenceTransformerTrainingArguments
from sentence_transformers.training_args import BatchSamplers

from sentence_transformers import SentenceTransformerTrainer
import json

dataset = load_dataset("meandyou200175/dataset_full_fixed", split="train")
dataset = dataset.rename_column("pos", "positive")
dataset = dataset.rename_column("neg", "negative")

dataset = dataset.train_test_split(test_size=0.1)

train_dataset = dataset['train']

eval_dataset = dataset['test']

train_dataset = train_dataset.train_test_split(test_size= 1 / 9)['train']
test_dataset = train_dataset.train_test_split(test_size= 1 / 9)['test']

corpus = {}
queries = {}
relevant_docs = {}

for idx, data in enumerate(eval_dataset):
    query_id = f"{2*idx}"
    positive_id = f"{2*idx+1}"

    corpus[positive_id] = data['positive']

    
    queries[query_id] = data['query']
    
    relevant_docs[query_id] = [positive_id]

ir_evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name=f"dim_{768}",
    truncate_dim=768, 
    score_functions={"cosine": cos_sim},
)
# model_id = 'intfloat/multilingual-e5-large'

# model = SentenceTransformer(
#     model_id,
#     device="cuda" if torch.cuda.is_available() else "cpu",
#     model_kwargs={"attn_implementation": "sdpa"},
# )
transformer = models.Transformer("vinai/phobert-base-v2", max_seq_length=256,)
pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean", 
                                pooling_mode_mean_tokens=True,
                                pooling_mode_cls_token=False,
                                pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[transformer, pooling],)
inner_train_loss = MultipleNegativesRankingLoss(model)

# print(ir_evaluator(model))

args = SentenceTransformerTrainingArguments(
    output_dir="/mnt/data1tb/thangcn/datnv2/models/embed/phobert-base-v2-tloss", 
    num_train_epochs=5,                       
    per_device_train_batch_size=60,            
    gradient_accumulation_steps=1,          
    per_device_eval_batch_size=60,            
    warmup_ratio=0.1,                          
    learning_rate=1e-6,                        
    lr_scheduler_type="constant_with_warmup",                
    optim="adamw_torch_fused",                 
    tf32=False,                            
    bf16=True,                                  
    batch_sampler=BatchSamplers.NO_DUPLICATES, 
    eval_strategy="epoch",                  
    save_strategy="epoch",                
    logging_steps=100,                       
    save_total_limit=3,               
    load_best_model_at_end=True,               
    metric_for_best_model="eval_dim_768_cosine_ndcg@10", 
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args, 
    train_dataset=train_dataset.select_columns(
        ["query", "positive"]
    ),
    loss= inner_train_loss,
    evaluator= ir_evaluator
)

trainer.train()

trainer.save_model()
trainer.push_to_hub('phobert-base-v2-tloss')
fine_tuned_model = SentenceTransformer(
    args.output_dir, device="cuda" if torch.cuda.is_available() else "cpu"
)

result2 = ir_evaluator(fine_tuned_model)
print(result2)
