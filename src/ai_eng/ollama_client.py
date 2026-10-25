"""Ollama (local model) implementation of the `LLMClient` protocol.

Uses Ollama's native /api/chat endpoint, not its Anthropic-compatible one, because the
compatible endpoint has no token-counting endpoint and no prompt caching.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import BaseModel

from ai_eng.llm import Message

DEFAULT_URL = "http://localhost:11434"
DEFAULT_MODEL = "phi4-mini"


class _ReplyMessage(BaseModel):
    content: str = ""


class _ChatResponse(BaseModel):
    message: _ReplyMessage = _ReplyMessage()
    prompt_eval_count: int = 0
    eval_count: int = 0
    eval_duration: int = 0  # nanoseconds
    load_duration: int = 0  # nanoseconds
    total_duration: int = 0  # nanoseconds


@dataclass(frozen=True)
class LocalUsage:
    prompt_tokens: int
    completion_tokens: int
    load_seconds: float
    total_seconds: float
    tokens_per_second: float


class OllamaClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        num_ctx: int = 4096,
        base_url: str = DEFAULT_URL,
        client: httpx.Client | None = None,
        timeout: float = 300.0,
    ) -> None:
        self.model = model
        self.num_ctx = num_ctx
        self._http = client or httpx.Client(base_url=base_url, timeout=timeout)
        self.last_usage: LocalUsage | None = None

    def _chat(self, messages: Sequence[Message], **options: Any) -> _ChatResponse:
        payload = {
            "model": self.model,
            "messages": [m.model_dump() for m in messages],
            "stream": False,
            "options": {"num_ctx": self.num_ctx, **options},
        }
        try:
            response = self._http.post("/api/chat", json=payload)
        except httpx.ConnectError as exc:
            raise RuntimeError("Cannot reach Ollama. Is it running? Try: ollama serve") from exc
        if response.status_code == 404:
            raise RuntimeError(f"Model '{self.model}' not found. Run: ollama pull {self.model}")
        response.raise_for_status()
        return _ChatResponse.model_validate(response.json())

    def complete(self, messages: Sequence[Message]) -> str:
        result = self._chat(messages)
        seconds = result.eval_duration / 1e9
        self.last_usage = LocalUsage(
            prompt_tokens=result.prompt_eval_count,
            completion_tokens=result.eval_count,
            load_seconds=result.load_duration / 1e9,
            total_seconds=result.total_duration / 1e9,
            tokens_per_second=result.eval_count / seconds if seconds else 0.0,
        )
        return result.message.content

    def count_tokens(self, messages: Sequence[Message]) -> int:
        """Ollama has no count endpoint: generate one token and read the prompt count."""
        return self._chat(messages, num_predict=1).prompt_eval_count
