from FlagEmbedding import FlagReranker
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,

)
from sentence_transformers.losses import MultipleNegativesRankingLoss, TripletLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator, RerankingEvaluator
reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)


# reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
import time

start = time.time()
# You can map the scores into 0-1 by set "normalize=True", which will apply sigmoid function to the score
# print(test_dataset['corpus']["1"])
scores = reranker.compute_score([[' đã được tiền huấn luyện trên bộ dữ liệu về y tế ', 'Retriever đau bụng bao gồm mô hình PhoBERT-base-v2 đã được tiền huấn luyện trên bộ dữ liệu về y tế - sức khỏe, mô hình bkai-foundation-models/vietnamese-bi-encoder và mô hình multilingual-e5-base đã được tinh chỉnh trên tác vụ truy xuất thông tin về y tế - sức khỏ']], normalize=True)
end = time.time()
print(scores) # [0.00027803096387751553, 0.9948403768236574]
print(end - start)
import torch
import json
embedder = SentenceTransformer('thang1943/bge-m3-finetuned')

with open('/mnt/data1tb/thangcn/datnv2/data/alobs_test_dataset.json', encoding='utf8') as f:
    test_dataset = json.load(f)
corpus = []
corpus_ids = []
for id in test_dataset['corpus'].keys():
    corpus.append(test_dataset['corpus'][id])
    corpus_ids.append(id)

corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
queries = []
query_ids = []
for id in test_dataset['queries'].keys():
    queries.append(test_dataset['queries'][id])
    query_ids.append(id)

# import torch
# import json
# embedder = SentenceTransformer('meandyou200175/paraphrase-multilingual-mpnet-base-v2_finetune_med')

# with open('alobs_test_dataset.json', encoding='utf8') as f:
#     test_dataset = json.load(f)
# corpus = []
# corpus_ids = []
# for id in test_dataset['corpus'].keys():
#     corpus.append(test_dataset['corpus'][id])
#     corpus_ids.append(id)

# corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
# queries = []
# query_ids = []
# for id in test_dataset['queries'].keys():
#     queries.append(test_dataset['queries'][id])
#     query_ids.append(id)

def find_similar_queries(queries, query_ids, corpus_embeddings, corpus_ids, embedder, k):
    top_k = k
    results = {}
    results_rerank = {}
    ii=0
    for query, query_id in zip(queries, query_ids):
        ii=ii+1
        if(ii % 100 == 0):
            print(ii,"/",len(queries))
        query_embedding = embedder.encode(query, convert_to_tensor=True)
        similarity_scores = embedder.similarity(query_embedding, corpus_embeddings)[0]
        scores, indices = torch.topk(similarity_scores, k=k)
        results[query_id] = [corpus_ids[idx] for idx in indices]
        reranked_indices = []
        inp = []
        for idx in results[query_id]:
            passage = test_dataset['corpus'][idx]
            inp.append([query, passage])
        score = reranker.compute_score(inp)
        for i, idx in zip(range(len(results[query_id])),results[query_id]):
            reranked_indices.append((idx, score[i]))
        # reranked_indices.append((idx, score))

        # Sort reranked indices based on scores
        reranked_indices.sort(key=lambda x: x[1], reverse=True)
        reranked_indices = [idx for idx, score in reranked_indices[:top_k]]
        results_rerank[query_id] = reranked_indices
    return results_rerank

def compute_accuracy(ground_truth_id, result):
    return 1 if ground_truth_id in result else 0

def compute_mrr(ground_truth_id, result):
    for rank, id in enumerate(result):
        if id == ground_truth_id:
            return 1 / (rank + 1)
    return 0

def calculate_metrics_for_query(ground_truth_ids, result):
    accuracy = compute_accuracy(ground_truth_ids, result)
    mrr = compute_mrr(ground_truth_ids, result)
    
    return accuracy, mrr

def calculate_metrics(queries, query_ids, test_dataset, results):
    sum_acc = 0
    sum_mrr = 0
    
    for query_id in query_ids:
        result = results[query_id]
        for i  in test_dataset['rel_ids'][query_id]:
            ground_truth_ids = i
        accuracy, mrr  = calculate_metrics_for_query(ground_truth_ids, result)
        
        sum_acc += accuracy
        sum_mrr += mrr
    num_queries = len(queries)
    metrics = {
        'accuracy': sum_acc / num_queries,
        'mrr': sum_mrr / num_queries,
    }
    
    return metrics
results_10 = find_similar_queries(queries, query_ids, corpus_embeddings, corpus_ids, embedder,k=10)
metrics_10 = calculate_metrics(queries, query_ids, test_dataset, results_10)
results_100 = find_similar_queries(queries, query_ids, corpus_embeddings, corpus_ids, embedder,k=100)
# results_1 = find_similar_queries(queries, query_ids, corpus_embeddings, corpus_ids, embedder,k=1)

# metrics_1 = calculate_metrics(queries, query_ids, test_dataset, results_1)

metrics_100 = calculate_metrics(queries, query_ids, test_dataset, results_100)
print(metrics_10)
print(metrics_100)