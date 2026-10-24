"""Single-request streaming latency probe supporting live GPU and --no-gpu modes."""

import argparse
import asyncio
import json
import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from llm_serving_platform.common.config import load_dotenv_file  # noqa: E402

load_dotenv_file()
DEFAULT_PROMPT = "Giải thích ngắn gọn cơ chế Continuous Batching."


def _extract_delta_text(data_str: str) -> str:
    """Safely extract non-empty assistant content from an SSE JSON payload."""
    try:
        parsed = json.loads(data_str)
    except json.JSONDecodeError:
        return ""
    if not isinstance(parsed, dict):
        return ""
    choices = parsed.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            delta = first_choice.get("delta")
            if isinstance(delta, dict):
                content = delta.get("content")
                if isinstance(content, str):
                    return content
    return ""


async def _measure_async(
    base_url: str,
    model: str,
    prompt: str,
    no_gpu: bool = False,
) -> int:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": 100,
    }

    start_time = time.perf_counter()
    first_content_time: float | None = None
    content_chunks = 0

    if no_gpu:
        from llm_serving_platform.gateway.main import create_app, get_worker
        from tests.mocks import MockInferenceWorker

        mock_worker = MockInferenceWorker(
            canned_chunks=[
                "Continuous",
                " Batching",
                " cho phép",
                " ghép",
                " các",
                " request",
                " ở mức",
                " iteration.",
            ],
            chunk_delay_s=0.005,
        )
        app = create_app()
        app.dependency_overrides[get_worker] = lambda: mock_worker
        client_ctx = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://mock-gateway",
            timeout=60.0,
        )
        target_url = "http://mock-gateway/v1/chat/completions"
    else:
        client_ctx = httpx.AsyncClient(timeout=60.0)
        target_url = f"{base_url.rstrip('/')}/v1/chat/completions"

    async with client_ctx as client:
        async with client.stream("POST", target_url, json=payload) as response:
            if response.status_code != 200:
                body = await response.aread()
                print(
                    f"[Lỗi] HTTP {response.status_code}: {body.decode('utf-8', errors='replace')}",
                    file=sys.stderr,
                )
                return 1

            async for raw_line in response.aiter_lines():
                line = raw_line.strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break

                content = _extract_delta_text(data)
                if not content:
                    continue

                if first_content_time is None:
                    first_content_time = time.perf_counter()
                    ttft = (first_content_time - start_time) * 1000
                    print(f"[Dữ kiện] TTFT (Time to First Content): {ttft:.2f} ms")

                content_chunks += 1
                print(content, end="", flush=True)

    end_time = time.perf_counter()
    e2e = (end_time - start_time) * 1000
    print(f"\n[Dữ kiện] E2E Latency: {e2e:.2f} ms (Content chunks: {content_chunks})")
    if content_chunks > 1 and first_content_time is not None:
        inter_chunk_ms = ((end_time - first_content_time) / (content_chunks - 1)) * 1000
        print(f"[Dữ kiện] Inter-Content-Chunk Interval: {inter_chunk_ms:.2f} ms")
    return 0


def stream_and_measure(
    prompt: str = DEFAULT_PROMPT,
    no_gpu: bool = False,
    base_url: str | None = None,
    model: str | None = None,
) -> int:
    target_base = base_url or os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000")
    target_model = model or os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
    return asyncio.run(
        _measure_async(
            base_url=target_base,
            model=target_model,
            prompt=prompt,
            no_gpu=no_gpu,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure streaming TTFT and E2E latency.")
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt message to send (default: 1-sentence Vietnamese prompt).",
    )
    parser.add_argument(
        "--no-gpu",
        "--mock",
        dest="no_gpu",
        action="store_true",
        help="Run in-process with MockInferenceWorker without requiring a GPU/vLLM server.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000"),
        help="Target base URL (vLLM or Gateway).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"),
        help="Model identifier.",
    )
    args = parser.parse_args()
    sys.exit(
        stream_and_measure(
            prompt=args.prompt,
            no_gpu=args.no_gpu,
            base_url=args.base_url,
            model=args.model,
        )
    )


if __name__ == "__main__":
    main()
