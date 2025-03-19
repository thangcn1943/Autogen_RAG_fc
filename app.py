import torch
import time
import streamlit as st
import os
from groq import Groq
from dotenv import load_dotenv
import json
from service.func_for_fc import rag_doctor_info, rag_medical, rag_price, book_appointment
import streamlit as st
from openai import OpenAI
import logging
logging.disable(logging.WARNING)

load_dotenv('.env')
groq_api_key = os.getenv("GROQ_API_KEY")
# client = Groq(api_key=groq_api_key)
# MODEL = os.getenv('MODEL')
open_ai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key = open_ai_key)
MODEL = 'gpt-4o'
EMBED_MODEL = os.getenv("EMBED_MODEL")

with open('/mnt/data1tb/thangcn/datnv2/prompts/tools.json', 'r') as f:
    tools = json.load(f)

def run_conversation(user_prompt):
    messages = [
        {
            "role": "system",
            "content": 'Xin chào! Tôi là trợ lý AI ve y te. Toi co the giai dap cac thac mac xung quanh benh vien va y te, hoac tao lich kham. Chuc ban mot ngay tot lanh!!!'
        },
        {
            "role": "user",
            "content": user_prompt,
        }
    ]
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=4096
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        # try:
        available_functions = {
            tool['function']['name']: globals()[tool['function']['name']] for tool in tools
        }

        messages.append(response_message)
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            print(function_name)
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(**function_args)
            
            return function_response
   
    else:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=4096
        )
        messages.append(response_message)
        final_response = response.choices[0].message.content

        return final_response

def main():

    st.title("THANGCN's AI Assistant")

    # Khởi tạo session state nếu chưa có
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Hiển thị lịch sử hội thoại
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Enter your query: ")

    start = time.time()
    if query:

        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        answer = run_conversation(query)

        with st.chat_message("assistant"):
            full_res = ""
            holder = st.empty()
            for word in answer.split():
                full_res += word + " "
                time.sleep(0.1)
                holder.markdown(full_res + "▌")
            holder.markdown(full_res)

        st.session_state.messages.append({"role": "assistant", "content": full_res})

        end = time.time()
        print("Time to process query:", end - start)

    else:
        print("Please enter your query")
    end = time.time()
    print("Time to process query: ", end-start)


if __name__ == "__main__":
    main()