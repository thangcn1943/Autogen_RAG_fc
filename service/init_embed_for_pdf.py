from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
import torch
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore


load_dotenv()
EMBED_MODEL = os.getenv("EMBED_MODEL") # 'BAAI/bge-small-en-v1.5' 
FAISS_ROOT = os.getenv("FAISS_ROOT") # 'faiss_db'

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={'device': 'cpu'}
)
os.makedirs(FAISS_ROOT, exist_ok=True)

def download_pdf(id: str):
    pass

def pdf_to_text(pdf_path: str, chunk_size = 1000):
    loader = DirectoryLoader(
        path = pdf_path,
        glob = "*.pdf",
        loader_cls = PyPDFLoader
    )
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=0,
        length_function=len
    )

    texts = text_splitter.split_documents(documents)

    return texts

async def store_embedding(pdf_path, chunk_size):
    texts = pdf_to_text(pdf_path, chunk_size)

    index = faiss.IndexFlatL2(len(embeddings.embed_query('thang')))
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        index_to_docstore_id={},
        docstore=InMemoryDocstore()
    )
    vector_store.add_documents(texts)
    vector_store.save_local(os.path.join(FAISS_ROOT, pdf_path.split('/')[-1]))

    print("Store embedding successfully!")
    torch.cuda.empty_cache()

# def main():
#     pdf_path = 
