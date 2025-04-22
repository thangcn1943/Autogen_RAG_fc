vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json \
    --chat-template "/mnt/data1tb/thangcn/datnv2/examples/tool_chat_template_llama3.1_json.jinja" \
    --max-model-len 8192