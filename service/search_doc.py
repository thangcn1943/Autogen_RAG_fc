from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from langchain.retrievers import EnsembleRetriever  
from langchain_community.retrievers import BM25Retriever  
from langchain_core.documents import Document  
from dotenv import load_dotenv
import os
load_dotenv('/mnt/data1tb/thangcn/datnv2/.env')

class ReRankerRetriever(BaseRetriever):
    def __init__(self, ensemble_retriever: EnsembleRetriever, reranker: CrossEncoder, top_k: int = 20, rerank_k: int = 10):
        super().__init__()
        self.ensemble_retriever = ensemble_retriever  # Đã sửa dấu phẩy thừa
        self.reranker = reranker
        self.top_k = top_k
        self.rerank_k = rerank_k 
    
    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        # Lấy documents từ ensemble retriever
        docs = self.ensemble_retriever.get_relevant_documents(query, top_k=self.top_k)
        
        # Tạo cặp (query, document) để re-rank
        pairs = [(query, doc.page_content) for doc in docs]
   
        # Tính điểm re-ranking
        rerank_scores = self.reranker.predict(pairs)
 
        # Gán điểm vào metadata
        for doc, score in zip(docs, rerank_scores):
            doc.metadata["rerank_score"] = float(score)  # Chuyển sang float để JSON serializable
    
        # Sắp xếp và trả về top documents
        sorted_docs = sorted(docs, key=lambda x: x.metadata["rerank_score"], reverse=True)
        return sorted_docs[:self.rerank_k]

def hybrid_search(vectorstore, query: str, k: int) -> ReRankerRetriever:
    """Create a hybrid retriever with re-ranking"""
    # Vector retriever
    retriever_vectordb = vectorstore.as_retriever(
        search_kwargs={"k": min(10, vectorstore.index.ntotal)}
    )
    
    # Keyword retriever
    documents = [
        Document(page_content=doc.page_content, metadata=doc.metadata)
        for doc in vectorstore.similarity_search(query, k=min(10, vectorstore.index.ntotal))
    ]
    keyword_retriever = BM25Retriever.from_documents(documents)
    keyword_retriever.k = k

    # Khởi tạo cross-encoder
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Tạo và trả về retriever với re-ranking
    return ReRankerRetriever(
        ensemble_retriever=EnsembleRetriever(
            retrievers=[retriever_vectordb, keyword_retriever],
            weights=[0.8, 0.2]
        ),
        reranker=cross_encoder,
        top_k=k*3,
        rerank_k=k   
    )