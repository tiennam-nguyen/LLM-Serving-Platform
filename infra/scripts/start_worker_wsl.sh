#!/usr/bin/env bash
uv run python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --dtype float16 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 2048 \
    --swap-space 1 \
    --host 0.0.0.0 \
    --port 8000