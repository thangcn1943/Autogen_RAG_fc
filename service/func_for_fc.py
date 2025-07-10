from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
from service.search_doc import hybrid_search
from openai import OpenAI
from service.generate_with_flux import prompt_to_image
from deep_translator import GoogleTranslator

load_dotenv('/mnt/data1tb/thangcn/datnv2/.env')
# Lấy các khóa API và mô hình
open_ai_key = os.getenv("OPENAI_API_KEY")
MODEL = 'gpt-4o' #os.getenv("MODEL", "gpt-4o")
EMBED_MODEL = "thang1943/multilingual-e5-large-v2" #os.getenv("EMBED_MODEL", "nampham1106/bkcare-embedding")
client = OpenAI(
    api_key = open_ai_key
)
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={'device': 'cpu'}
)

def rag_service_info(query: str):
    service_info = FAISS.load_local('/mnt/data1tb/thangcn/datnv2/vector_database/FAISS_2/service_info', embeddings, allow_dangerous_deserialization=True)
    ensemble_retriever = hybrid_search(service_info,query,10)
    return ensemble_retriever


def rag_product_info(query: str):
    product_info = FAISS.load_local('/mnt/data1tb/thangcn/datnv2/vector_database/FAISS_2/product_json_info', embeddings, allow_dangerous_deserialization=True)
    ensemble_retriever = hybrid_search(product_info,query,10)
    return ensemble_retriever

def rag_doctor_info(query: str):
    doctor_info = FAISS.load_local('/mnt/data1tb/thangcn/datnv2/vector_database/FAISS_2/doctor_info', embeddings, allow_dangerous_deserialization=True)
    ensemble_retriever = hybrid_search(doctor_info,query,10)
    return ensemble_retriever

def qa_medical(query: str):
    # Thêm post retrieval
    # response = client.chat.completions.create(
    #     model = MODEL,
    #     messages = [
    #         {'role': 'system', 'content': 'As a medical expert, please provide a detailed response to the following question'},
    #         {'role': 'user', 'content': f"{query}"}
    #     ],
    #     max_tokens = 512
    # ).choices[0].message.content

    result = GoogleTranslator(source='vi', target='en').translate(query)
    image = prompt_to_image(result, save_previews=True)
    # print(type(image))
    medical_info = FAISS.load_local('/mnt/data1tb/thangcn/datnv2/vector_database/FAISS_2/qa_document', embeddings, allow_dangerous_deserialization=True)
    ensemble_retriever = hybrid_search(medical_info,query,10)
    # print('-' * 50)
    # print(type(ensemble_retriever))
    return ensemble_retriever, image

def qa_symptom(query: str):
    # Thêm post retrieval
    # response = client.chat.completions.create(
    #     model = MODEL,
    #     messages = [
    #         {'role': 'system', 'content': 'As a medical expert specializing in diagnosing diseases based on clinical signs, please provide an answer to the following question.'},
    #         {'role': 'user', 'content': f"{query}"}
    #     ],
    #     max_tokens = 512
    # ).choices[0].message.content

    result = GoogleTranslator(source='vi', target='en').translate(query)
    image = prompt_to_image(result, save_previews=True)
    # print(type(image))
    medical_info = FAISS.load_local('/mnt/data1tb/thangcn/datnv2/vector_database/FAISS_2/symptoms', embeddings, allow_dangerous_deserialization=True)
    ensemble_retriever = hybrid_search(medical_info,query,10)
    # print('-' * 50)
    # print(type(ensemble_retriever))
    return ensemble_retriever, image


def book_appointment(name=None, phone=None, date=None, time=None, specialty=None):
    # Danh sách các tham số bắt buộc và mô tả
    required_params = {
        "name": "Tên người đặt hẹn",
        "phone": "Số điện thoại",
        "date": "Ngày hẹn (YYYY-MM-DD)",
        "time": "Giờ hẹn (HH:MM)",
        "specialty": "Chuyên khoa"
    }
    
    # Kiểm tra tham số thiếu
    missing_params = [
        param_name for param_name, param_value in {
            "name": name,
            "phone": phone,
            "date": date,
            "time": time,
            "specialty": specialty
        }.items() if param_value is None or param_value == "" or param_value == "NaN"
    ]

    
    # Nếu thiếu tham số, trả về yêu cầu điền thông tin
    if missing_params:
        missing_descriptions = [required_params[param] for param in missing_params]
        return f"⚠️ Vui lòng cung cấp: {', '.join(missing_descriptions)}"
    
    # Kiểm tra định dạng số điện thoại
    if not phone.isdigit() or len(phone) < 10:
        return "⚠️ Số điện thoại không hợp lệ (phải là số và ít nhất 10 chữ số)"
    
    # Kiểm tra định dạng ngày và giờ (có thể thêm regex chi tiết)
    if len(date) != 10 or date.count("-") != 2:
        return "⚠️ Định dạng ngày không hợp lệ (YYYY-MM-DD)"
    if len(time) != 5 or time.count(":") != 1:
        return "⚠️ Định dạng giờ không hợp lệ (HH:MM)"
    
    # Trả về thông tin đặt hẹn nếu hợp lệ
    appointment_details = {
        "name": name,
        "phone": phone,
        "date": date,
        "time": time,
        "specialty": specialty,
        "status": "Đã xác nhận"
    }
    
    return f"🗓️ Đặt hẹn thành công với thông tin"


def merge_faiss_stores(query, index_paths):
    if not index_paths:
        return None
    
    # Load index đầu tiên làm base
    merged_store = FAISS.load_local(
        index_paths[0], 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    # Merge các index còn lại
    for path in index_paths[1:]:
        store = FAISS.load_local(
            path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        merged_store.merge_from(store)
    ensemble_retriever = hybrid_search(merged_store,query,10)
    return ensemble_retriever
