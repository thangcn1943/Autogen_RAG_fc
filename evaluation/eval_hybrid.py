import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document  
from langchain_community.retrievers import BM25Retriever  
from sentence_transformers import CrossEncoder

with open('/mnt/data1tb/thangcn/datnv2/data/alobs_test_dataset.json', encoding='utf8') as f:
    test_dataset = json.load(f)

corpus = list(test_dataset['corpus'].values())
corpus_ids = list(test_dataset['corpus'].keys())

queries = list(test_dataset['queries'].values())
query_ids = list(test_dataset['queries'].keys())

documents = [
    Document(page_content=text, metadata={"doc_id": doc_id})
    for text, doc_id in zip(corpus, corpus_ids)
]

# Danh sách các mô hình embedding cần đánh giá
models = [
    'thang1943/multilingual-e5-large-v2',
]

def compute_accuracy(gt_ids, result_ids):
    return int(any(gt_id in result_ids for gt_id in gt_ids))

def compute_mrr(gt_ids, result_ids):
    for rank, doc_id in enumerate(result_ids):
        if doc_id in gt_ids:
            return 1 / (rank + 1)
    return 0

rerank_retriever = CrossEncoder("BAAI/bge-reranker-v2-m3")

def find_top_k(queries, query_ids, retriever):
    results = {}
    for query, qid in zip(queries, query_ids):
        retrieved_docs = retriever.get_relevant_documents(query)

        pairs = [(query, doc.page_content) for doc in retrieved_docs]
        rerank_scores = rerank_retriever.predict(pairs)

        scored_docs = list(zip(retrieved_docs, rerank_scores))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        results[qid] = [doc.metadata['doc_id'] for doc, score in scored_docs]
    return results

# Lặp qua các mô hình embedding
for model in models:
    print(f"\n🔍 Đang đánh giá mô hình: {model}")

    # Tạo hàm embedding LangChain
    embeddings = HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"device": "cuda"}
    )

    # Tạo FAISS index
    dummy_vector = embeddings.embed_query("test")
    dim = len(dummy_vector)
    index = faiss.IndexFlatL2(dim)

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        index_to_docstore_id={},
        docstore=InMemoryDocstore()
    )
    vector_store.add_documents(documents)

    # Tạo retriever vector (dense)
    retriever_vector = vector_store.as_retriever(search_kwargs={"k": 10})

    # Tạo retriever BM25 (sparse)
    retriever_bm25 = BM25Retriever.from_documents(documents)
    retriever_bm25.k = 10

    # Kết hợp retriever (Hybrid)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever_vector, retriever_bm25],
        weights=[0.8, 0.2]  # Có thể điều chỉnh
    )

    # Truy xuất kết quả
    results = find_top_k(queries, query_ids, ensemble_retriever)

    # Đánh giá
    total_acc = 0
    total_mrr = 0
    for qid in query_ids:
        gt_ids = test_dataset['rel_ids'][qid]
        result_ids = results[qid]
        # total_acc += compute_accuracy(gt_ids, result_ids)
        total_mrr += compute_mrr(gt_ids, result_ids)

    num_queries = len(query_ids)
    # print(f"✅ Accuracy: {total_acc / num_queries:.4f}")
    print(f"✅ MRR     : {total_mrr / num_queries:.4f}")
    print('-' * 80)