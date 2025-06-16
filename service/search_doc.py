from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever  
from langchain_core.documents import Document  
from dotenv import load_dotenv
import os
from typing import List

load_dotenv('/mnt/data1tb/thangcn/datnv2/.env')

class RerankRetriever():
    def __init__(self, ensemble_retriever, rerank_retriever, top_k = 30, rerank_k = 10):
        self.ensemble_retriever = ensemble_retriever
        self.rerank_retriever = rerank_retriever
        self.top_k = top_k
        self.rerank_k = rerank_k
    
    def get_relevant_documents(self, query: str):
        docs = self.ensemble_retriever.get_relevant_documents(query)[:self.top_k]
        # print(f"Initial docs: {len(docs)}")  # Kiểm tra số lượng docs
        
        pairs = [(query, doc.page_content) for doc in docs]
        # print(f"First pair sample: {pairs[0][1][:50]}...")  # Xem nội dung có hợp lệ
        
        rerank_scores = self.rerank_retriever.predict(pairs)
        # print(f"Scores: {rerank_scores}")  # Kiểm tra điểm re-rank
        
        for doc, score in zip(docs, rerank_scores):
            doc.metadata["rerank_score"] = float(score)
        
        sorted_docs = sorted(docs, key=lambda x: x.metadata["rerank_score"], reverse=True)[:self.rerank_k]
        # print(f"Final docs: {[d.metadata['rerank_score'] for d in sorted_docs]}")  # Xem điểm cuối cùng
        return sorted_docs


def hybrid_search(vectorstore, query: str, k: int) -> RerankRetriever:
    """Create a hybrid retriever with re-ranking"""
    # Vector retriever
    retriever_vectordb = vectorstore.as_retriever(
        search_kwargs={"k": min(k, vectorstore.index.ntotal)}
    )
    
    # Keyword retriever
    documents = [
        Document(page_content=doc.page_content, metadata=doc.metadata)
        for doc in vectorstore.similarity_search(query, k=min(k, vectorstore.index.ntotal))
    ]
    keyword_retriever = BM25Retriever.from_documents(documents)
    keyword_retriever.k = k

    # Create ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever_vectordb, keyword_retriever],
        weights=[0.8, 0.2]
    )

    # Initialize cross-encoder
    cross_encoder = CrossEncoder("BAAI/bge-reranker-v2-m3")

    return RerankRetriever(
        ensemble_retriever= ensemble_retriever,
        rerank_retriever=cross_encoder,
        top_k=30,
        rerank_k=k
    )