vllm serve Qwen/Qwen3-8B \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --chat-template "/mnt/data1tb/thangcn/datnv2/chat_template/qwen3-8B-template.jinja" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.8