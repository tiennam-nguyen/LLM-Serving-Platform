import json
import os
import time

import requests

BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.1.1:8000")

def stream_and_measure():
    payload = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "messages": [
            {"role": "user", "content": "Giải thích ngắn gọn cơ chế Continuous Batching."}
        ],
        "stream": True,
        "max_tokens": 100
    }

    start_time = time.perf_counter()
    first_token_time = None
    token_count = 0

    # Kích hoạt kết nối SSE
    response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, stream=True)

    for chunk in response.iter_lines():
        if chunk:
            data = chunk.decode("utf-8").removeprefix("data: ")
            if data == "[DONE]":
                break

            if first_token_time is None:
                first_token_time = time.perf_counter()
                ttft = (first_token_time - start_time) * 1000
                print(f"[Dữ kiện] TTFT (Time to First Token): {ttft:.2f} ms")

            token_count += 1
            # In ra để xác minh luồng stream đang chạy
            print(json.loads(data)["choices"][0]["delta"].get("content", ""), end="", flush=True)

    end_time = time.perf_counter()
    if token_count > 1:
        tpot = ((end_time - first_token_time) / (token_count - 1)) * 1000
        print(f"\n[Dữ kiện] TPOT (Time Per Output Token): {tpot:.2f} ms")

if __name__ == "__main__":
    stream_and_measure()
