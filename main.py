from typing import List, Dict, Any
from langchain_core.runnables.history import RunnableWithMessageHistory
from prompts.prompt import contextualize_q_system_prompt, qa_system_prompt
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_openai import ChatOpenAI
from service.func_for_fc import rag_doctor_info, rag_product_info, rag_service_info
import os
import json
import time
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory 
import uuid
import atexit

from service.message_stored import load_session_history, get_db, save_message

store = {}
session_id = 'thangcn19'
print(session_id)
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = load_session_history(session_id)
    return store[session_id]

def save_all_sessions():
    for session_id, chat_history in store.items():
        for message in chat_history.messages:
            save_message(session_id, message["role"], message["content"])
atexit.register(save_all_sessions)

load_dotenv('/mnt/data1tb/thangcn/datnv2/.env')
open_ai_key = os.getenv("OPENAI_API_KEY")
MODEL = 'gpt-4o' #os.getenv("MODEL", "gpt-4o")
EMBED_MODEL = "nampham1106/bkcare-embedding" #os.getenv("EMBED_MODEL", "nampham1106/bkcare-embedding")

with open('/mnt/data1tb/thangcn/datnv2/prompts/tools.json', 'r') as f:
    function_schema = json.load(f)

llm = ChatOpenAI(model=MODEL, temperature=0, api_key=open_ai_key)

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
    return response

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

    if user_prompt := st.chat_input("Enter your user_prompt: "):
        start_time = time.time()
        
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)
        r = process_llm_function_call(load_session_history(session_id), user_prompt)
        answer = None
        Is_call_function = False
        # Kiểm tra xem có function_call trong response không
        if 'function_call' in r.additional_kwargs:
            available_functions = {tool['name']: globals()[tool['name']] 
                                for tool in function_schema}
            function_args = json.loads(r.additional_kwargs['function_call']['arguments'])
            function_name = r.additional_kwargs['function_call']['name']
            function_to_call = available_functions.get(function_name)
            function_response = function_to_call(**function_args)
            Is_call_function = True
        else:
            # Xử lý trường hợp không có function call
            print("Không có function call trong response")
            function_response = r.content
        if Is_call_function:
        # Prompt để ngữ cảnh hóa câu hỏi

            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [("system", contextualize_q_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
            )
            history_aware_retriever = create_history_aware_retriever(llm, function_response, contextualize_q_prompt)

            qa_prompt = ChatPromptTemplate.from_messages(
                [("system", qa_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
            )

            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


            store = {}
            def get_session_history(session_id: str) -> BaseChatMessageHistory:
                if session_id not in store:
                    store[session_id] = ChatMessageHistory()
                return store[session_id]

            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )

            history_conversation = f"""
                History conversation:\n
                """
            for msg in load_session_history(session_id).messages[max(-len(load_session_history(session_id).messages), -5):]:
                history_conversation += f"'role' : '{msg['role']}' , 'content' : ' {msg['content']}' \n"

            def invoke_and_save(session_id, input_text):

                # Save the user question with role "human"
                save_message(session_id, "user", input_text)
                
                result = conversational_rag_chain.invoke(
                    {"input": history_conversation + '\nQuery: ' + input_text},
                    config={"configurable": {"session_id": session_id}}
                )["answer"]
                # Save the AI answer with role "ai"
                save_message(session_id, "assistant", result)
                return result

            answer = invoke_and_save(session_id, user_prompt)
        else:
            answer = function_response
        
        with st.chat_message("assistant"):
            full_response = stream_response(answer)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        print(f"Time to process query: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()