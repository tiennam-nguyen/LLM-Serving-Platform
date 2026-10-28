"""Pydantic schemas for the API Gateway (compatible with Pydantic v2 and v1)."""

from typing import Any, Dict, List, Optional

import pydantic
from pydantic import BaseModel, Field

_PYDANTIC_V2 = int(pydantic.__version__.split(".")[0]) >= 2

if _PYDANTIC_V2:
    from pydantic import field_validator
else:
    from pydantic import validator as _v1_validator


def _check_messages_list(v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not v:
        raise ValueError("messages list cannot be empty.")
    for idx, msg in enumerate(v):
        if not isinstance(msg, dict):
            raise ValueError(f"Message at index {idx} must be a dictionary.")
        if "role" not in msg or "content" not in msg:
            raise ValueError(f"Message at index {idx} must contain 'role' and 'content' keys.")
        if not isinstance(msg["role"], str) or not isinstance(msg["content"], str):
            raise ValueError(f"Message at index {idx} 'role' and 'content' must be strings.")
    return v


class ChatCompletionRequest(BaseModel):
    """Payload schema for POST /v1/chat/completions."""

    model: str = Field(..., min_length=1, description="Target LLM model ID")
    messages: List[Dict[str, Any]] = Field(
        ..., min_length=1 if _PYDANTIC_V2 else None, description="Conversation history messages"
    )
    stream: bool = Field(True, description="Whether to stream back SSE tokens")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(None, ge=1)

    if _PYDANTIC_V2:
        model_config = {"extra": "allow"}

        @field_validator("messages")
        @classmethod
        def validate_messages(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            return _check_messages_list(v)

    else:

        class Config:
            extra = "allow"

        @_v1_validator("messages", pre=True, always=True)
        def validate_messages(cls, v: Any) -> List[Dict[str, Any]]:
            if not isinstance(v, list):
                raise ValueError("messages must be a list.")
            return _check_messages_list(v)

        def model_dump(self, **kwargs: Any) -> Dict[str, Any]:
            return self.dict(**kwargs)
