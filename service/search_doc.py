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

        pairs = [(query, doc.page_content) for doc in docs]
    
        rerank_scores = self.rerank_retriever.predict(pairs)
        
        for doc, score in zip(docs, rerank_scores):
            doc.metadata["rerank_score"] = float(score)
        
        sorted_docs = sorted(docs, key=lambda x: x.metadata["rerank_score"], reverse=True)[:self.rerank_k]
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
    cross_encoder = CrossEncoder("BAAI/bge-reranker-v2-m3", device = "cpu")

    return RerankRetriever(
        ensemble_retriever= ensemble_retriever,
        rerank_retriever=cross_encoder,
        top_k=30,
        rerank_k=k
    )

