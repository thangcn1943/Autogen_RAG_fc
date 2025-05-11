vllm serve thang1943/Llama-3.1-8B-in-ViMed \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json \
    --chat-template "/mnt/data1tb/thangcn/datnv2/chat_template/tool_chat_template_llama3.1_json.jinja" \
    --max-model-len 8192