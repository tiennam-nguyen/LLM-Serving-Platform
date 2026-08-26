#!/usr/bin/env bash
# Khởi chạy vLLM worker cho môi trường WSL2 (giới hạn 4GB VRAM)
# Yêu cầu: Đã chạy `uv sync` và kích hoạt môi trường.

uv run python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --dtype float16 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 2048 \
    --port 8000