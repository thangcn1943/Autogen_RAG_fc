import os
from groq import Groq
import dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from service.search_doc import retrieve_and_re_rank_advanced
from openai import OpenAI
# MODEL = os.getenv('MODEL')
MODEL = 'gpt-4o'
EMBED_MODEL = os.getenv("EMBED_MODEL") # 'BAAI/bge-small-en-v1.5'
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={'device': 'cpu'}
)
dotenv.load_dotenv('/mnt/data1tb/thangcn/datnv2/.env') 
groq_api_key = os.getenv("GROQ_API_KEY")
open_ai_key = os.getenv("OPENAI_API_KEY")
# client = Groq(api_key=groq_api_key)
client = OpenAI(api_key = open_ai_key)

def book_appointment(query: str):
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": 
                                    """Bạn là một trợ lí ảo hỗ trợ trích xuất thông tin về lịch hẹn khám bệnh, hãy trích xuất các thông tin dưới đây về thành dạng json(yêu cầu phải có các thông tin về họ và tên, số điện thoại liên hệ, ngày giờ khám bệnh, chuyên khoa khám bệnh, bác sĩ nếu yêu cầu(nếu k có bác sĩ thì khi là k yêu cầu)), nếu thiếu bất kì thông tin gì thì hãy yêu cầu nhập thêm. ví dụ dưới đây
                                    Query: Tôi là Thắng và muốn khám bệnh về tiểu đường, sđt 0123456789, 15h chiều ngày 21-3-2025
                                    Response:
                                    {
                                        “name”: “Thang”,
                                        “phone_number”: “0123456789”,
                                        “date_time”: “15:00 21-3-2025”,
                                        “specialize”: “Nội tiết”,
                                        “doctor”: “not required”

                                    }"""},
            {"role": "user", "content": f"{query}"},
        ],
        model= MODEL,
    )
    return chat_completion.choices[0].message.content
def rag_price(question: str):
    vector_db = FAISS.load_local('/mnt/data1tb/datn/faiss_db/price', embeddings, allow_dangerous_deserialization=True)
    retrieved_docs, scores = retrieve_and_re_rank_advanced(vector_db, question)
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant that utilizes retrieved information."},
            {"role": "user", "content": f"Context:\n{retrieved_docs}\n\nQuestion: {question}"},
        ],
        model= MODEL,
    )
    # print(chat_completion.choices[0].message.content)
    return chat_completion.choices[0].message.content

def rag_doctor_info(question: str):
    vector_db = FAISS.load_local('/mnt/data1tb/datn/faiss_db/doctor_info', embeddings, allow_dangerous_deserialization=True)
    retrieved_docs, scores = retrieve_and_re_rank_advanced(vector_db, question)
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant that utilizes retrieved information."},
            {"role": "user", "content": f"Context:\n{retrieved_docs[0]}\n\nQuestion: {question}"},
        ],
        model= MODEL,
    )
    # print(chat_completion.choices[0].message.content)
    return chat_completion.choices[0].message.content

def rag_medical(question: str):
    vector_db = FAISS.load_local('/mnt/data1tb/datn/faiss_db/pdf_medical', embeddings, allow_dangerous_deserialization=True)
    retrieved_docs, scores = retrieve_and_re_rank_advanced(vector_db, question)
        # Gửi yêu cầu đến mô hình Groq
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant that utilizes retrieved information."},
            {"role": "user", "content": f"Context:\n{retrieved_docs[0]}\n\nQuestion: {question}"},
        ],
        model= MODEL,
    )
    # print(chat_completion.choices[0].message.content)
    return chat_completion.choices[0].message.content