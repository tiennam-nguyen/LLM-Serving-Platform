"""Centralized configuration and .env loader for the LLM Serving Platform."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_ENV_FILE = REPO_ROOT / ".env"


def load_dotenv_file(dotenv_path: Path = DEFAULT_ENV_FILE, override: bool = False) -> None:
    """Load key=value pairs from a .env file into os.environ without external dependencies."""
    if not dotenv_path.is_file():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].strip()
        if key and (override or key not in os.environ):
            os.environ[key] = value


# Automatically load root .env file if present
load_dotenv_file()


def _parse_csv_env(env_key: str, default: List[str]) -> List[str]:
    raw = os.getenv(env_key, "").strip()
    if not raw:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_int_env(env_key: str, default: int) -> int:
    raw = os.getenv(env_key, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _parse_float_env(env_key: str, default: float) -> float:
    raw = os.getenv(env_key, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _parse_bool_env(env_key: str, default: bool = False) -> bool:
    raw = os.getenv(env_key, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on", "mock", "no-gpu", "no_gpu"}:
        return True
    if raw in {"0", "false", "no", "off", "gpu", "live"}:
        return False
    return default


@dataclass(frozen=True)
class GatewayConfig:
    """Immutable configuration for the Data-Plane API Gateway loaded from .env / environment."""

    env_mode: str = field(default_factory=lambda: os.getenv("ENV_MODE", "development"))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    gateway_host: str = field(default_factory=lambda: os.getenv("GATEWAY_HOST", "0.0.0.0"))
    gateway_port: int = field(default_factory=lambda: _parse_int_env("GATEWAY_PORT", 8080))
    vllm_base_url: str = field(
        default_factory=lambda: os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000")
    )
    worker_id: str = field(default_factory=lambda: os.getenv("WORKER_ID", "worker-vllm-0"))
    worker_timeout_s: float = field(
        default_factory=lambda: _parse_float_env("VLLM_TIMEOUT_S", 60.0)
    )
    model_name: str = field(
        default_factory=lambda: os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
    )
    no_gpu: bool = field(
        default_factory=lambda: (
            _parse_bool_env("NO_GPU", False)
            or os.getenv("WORKER_MODE", "").strip().lower()
            in {"mock", "no-gpu", "no_gpu", "cpu"}
        )
    )
    cors_allowed_origins: List[str] = field(
        default_factory=lambda: _parse_csv_env(
            "CORS_ALLOWED_ORIGINS",
            [
                "http://localhost:3000",
                "http://127.0.0.1:3000",
                "http://localhost:3001",
                "http://127.0.0.1:3001",
            ],
        )
    )
