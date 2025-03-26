import dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
from service.search_doc import hybrid_search


load_dotenv('/mnt/data1tb/thangcn/datnv2/.env')
# Lấy các khóa API và mô hình
open_ai_key = os.getenv("OPENAI_API_KEY")
MODEL = 'gpt-4o' #os.getenv("MODEL", "gpt-4o")
EMBED_MODEL = "nampham1106/bkcare-embedding" #os.getenv("EMBED_MODEL", "nampham1106/bkcare-embedding")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={'device': 'cpu'}
)

def rag_service_info(query: str):
    service_info = FAISS.load_local('/mnt/data1tb/thangcn/datnv2/vector_database/faiss/service_info', embeddings, allow_dangerous_deserialization=True)
    ensemble_retriever = hybrid_search(service_info,query,10)
    return ensemble_retriever


def rag_product_info(query: str):
    product_info = FAISS.load_local('/mnt/data1tb/thangcn/datnv2/vector_database/faiss/product_info', embeddings, allow_dangerous_deserialization=True)
    ensemble_retriever = hybrid_search(product_info,query,10)
    return ensemble_retriever

def rag_doctor_info(query: str):
    doctor_info = FAISS.load_local('/mnt/data1tb/thangcn/datnv2/vector_database/faiss/doctor_info', embeddings, allow_dangerous_deserialization=True)
    ensemble_retriever = hybrid_search(doctor_info,query,10)
    return ensemble_retriever
