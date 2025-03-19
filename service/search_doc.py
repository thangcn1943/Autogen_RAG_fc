from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from langchain.retrievers import EnsembleRetriever  
from langchain_community.retrievers import BM25Retriever  
from langchain_core.documents import Document  
from dotenv import load_dotenv
import os
load_dotenv()

def hybrid_search(vectorstore,query, k):
    retriever_vectordb = vectorstore.as_retriever(search_kwargs={"k": k})

    documents = [
        Document(page_content=doc.page_content, metadata=doc.metadata)
        for doc in vectorstore.similarity_search(query, k=100)
    ]
    keyword_retriever = BM25Retriever.from_documents(documents)
    keyword_retriever.k =  k

    ensemble_retriever = EnsembleRetriever(retrievers=[retriever_vectordb,keyword_retriever],weights=[0.5, 0.5])
    return ensemble_retriever

def retrieve_and_re_rank(vector_db, query, k=10):
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    ensemble_search = hybrid_search(vector_db,query, k)

    docs_rel = ensemble_search.get_relevant_documents(query)

    pairs = [[query, doc] for doc in docs_rel]

    rerank_scores = cross_encoder.predict(pairs)

    ranked_docs = sorted(zip(docs_rel, rerank_scores), key=lambda x: x[1], reverse=True)
    
    results = [doc for doc, _ in ranked_docs]
    scores = [score for _, score in ranked_docs]
    
    return results, scores
