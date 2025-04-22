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

def qa_medical(query: str):
    medical_info = FAISS.load_local('/mnt/data1tb/thangcn/datnv2/vector_database/faiss/qa_document', embeddings, allow_dangerous_deserialization=True)
    ensemble_retriever = hybrid_search(medical_info,query,10)
    return ensemble_retriever

def qa_symptom(query: str):
    medical_info = FAISS.load_local('/mnt/data1tb/thangcn/datnv2/vector_database/faiss/symptoms', embeddings, allow_dangerous_deserialization=True)
    ensemble_retriever = hybrid_search(medical_info,query,10)
    return ensemble_retriever

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
        }.items() if param_value is None
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