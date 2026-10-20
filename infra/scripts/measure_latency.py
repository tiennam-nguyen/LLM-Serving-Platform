import json
import os
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from src.llm_serving_platform.common.config import load_dotenv_file  # noqa: E402

load_dotenv_file()
BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000")


def stream_and_measure() -> None:
    payload = {
        "model": os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"),
        "messages": [
            {"role": "user", "content": "Giải thích ngắn gọn cơ chế Continuous Batching."}
        ],
        "stream": True,
        "max_tokens": 100,
    }

    start_time = time.perf_counter()
    first_token_time = None
    token_count = 0

    with httpx.Client(timeout=60.0) as client:
        with client.stream("POST", f"{BASE_URL}/v1/chat/completions", json=payload) as response:
            for raw_line in response.iter_lines():
                line = raw_line.strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break

                if first_token_time is None:
                    first_token_time = time.perf_counter()
                    ttft = (first_token_time - start_time) * 1000
                    print(f"[Dữ kiện] TTFT (Time to First Token): {ttft:.2f} ms")

                token_count += 1
                content = json.loads(data)["choices"][0]["delta"].get("content", "")
                print(content, end="", flush=True)

    end_time = time.perf_counter()
    if token_count > 1 and first_token_time is not None:
        tpot = ((end_time - first_token_time) / (token_count - 1)) * 1000
        print(f"\n[Dữ kiện] TPOT (Time Per Output Token): {tpot:.2f} ms")


if __name__ == "__main__":
    stream_and_measure()
