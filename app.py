from dotenv import load_dotenv
from prompts.prompt import qa_system_prompt, contextualize_q_system_prompt
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
import json
import time
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from service.search_doc import hybrid_search
import streamlit as st
from openai import OpenAI
import logging
import uuid
from service.func_for_fc import rag_service_price, rag_doctor_info, rag_product_price
from service.message_stored import save_message, load_session_history

logging.disable(logging.WARNING)

store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = load_session_history(session_id)
    return store[session_id]

# Ensure you save the chat history to the database when needed
def save_all_sessions():
    for session_id, chat_history in store.items():
        for message in chat_history.messages:
            save_message(session_id, message["role"], message["content"])

import atexit
atexit.register(save_all_sessions)

load_dotenv('/mnt/data1tb/thangcn/datnv2/.env')
# Lấy các khóa API và mô hình
open_ai_key = os.getenv("OPENAI_API_KEY")
MODEL = 'gpt-4o' #os.getenv("MODEL", "gpt-4o")
EMBED_MODEL = "nampham1106/bkcare-embedding" #os.getenv("EMBED_MODEL", "nampham1106/bkcare-embedding")

session_id = uuid.uuid4()


with open('/mnt/data1tb/thangcn/datnv2/prompts/tools.json', 'r') as f:
    function_schema = json.load(f)
session_id = str(uuid.uuid4())

llm = ChatOpenAI(model=MODEL, temperature=0, api_key=open_ai_key)
#Khoi tao prompt
def create_contextualize_prompt(contextualize_q_system_prompt, qa_system_prompt):
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt), 
            MessagesPlaceholder("chat_history"), 
            ("human", "{input}")
        ]
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [("system", qa_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
    )

    return contextualize_q_prompt, qa_prompt

contextualize_q_prompt, qa_prompt = create_contextualize_prompt(contextualize_q_system_prompt, qa_system_prompt)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)     

# su ly function calling
def process_llm_function_call(chat_history, user_prompt: str):
    messages = []
    for msg in chat_history.messages:
        messages.append(msg)
    # Thêm câu hỏi mới nhất
    messages.append(
        {"role": "user", "content": user_prompt}
    ) 
    # Gọi LLM với function calling
    response = llm.predict_messages(
        messages,
        functions=function_schema
    )
    print(response)
    return response

def execute_function_call(chat_history, user_prompt, function_schema: list):
    r = process_llm_function_call(chat_history, user_prompt)
    # if 'function_call' in r.additional_kwargs
    available_functions = {tool['name']: globals()[tool['name']] 
                            for tool in function_schema}
    function_args = json.loads(r.additional_kwargs['function_call']['arguments'])
    function_name = r.additional_kwargs['function_call']['name']
    query = function_args['query']
    print(function_name)
    function_to_call = available_functions.get(function_name)
    function_response = function_to_call(**function_args)
    return function_response, query

def process_user_query(chat_history, user_prompt: str, function_schema: list) -> str:
    
    function_response, query = execute_function_call(chat_history,user_prompt, function_schema)
    
    return _process_rag_chain(function_response, user_prompt, query)

# output va dau ra
def invoke_and_save(session_id,conversational_rag_chain, input_text, query):
    # Save the user question with role "human"
    save_message(session_id, "user", input_text)
    
    result = conversational_rag_chain.invoke(
        {"input": input_text},
        config={"configurable": {"session_id": session_id}}
    )["answer"]

    # Save the AI answer with role "ai"
    save_message(session_id, "assistant", result)
    return result

def _process_rag_chain(ensemble_retriever, user_prompt: str, query) -> str:
    history_aware_retriever = create_history_aware_retriever(
        llm, ensemble_retriever, contextualize_q_prompt
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    
    return invoke_and_save(session_id, conversational_rag_chain, user_prompt, query)

def display_chat_history():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

def stream_response(answer: str) -> str:
    full_res = ""
    holder = st.empty()
    for word in answer.split():
        full_res += word + " "
        time.sleep(0.1)
        holder.markdown(full_res + "▌")
    holder.markdown(full_res)
    return full_res

def main():
    st.title("THANGCN's AI Assistant")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    display_chat_history()

    if query := st.chat_input("Enter your query: "):
        start_time = time.time()
        
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        answer = process_user_query(load_session_history(session_id), query, function_schema)
        
        with st.chat_message("assistant"):
            full_response = stream_response(answer)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        print(f"Time to process query: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()