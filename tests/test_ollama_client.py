import json
from typing import Any

import httpx
import pytest

from ai_eng.llm import Message
from ai_eng.ollama_client import OllamaClient

REPLY: dict[str, Any] = {
    "message": {"role": "assistant", "content": "hi there"},
    "prompt_eval_count": 21,
    "eval_count": 4,
    "eval_duration": 2_000_000_000,
    "load_duration": 500_000_000,
    "total_duration": 3_000_000_000,
}


def make_client(status: int = 200) -> tuple[OllamaClient, list[dict[str, Any]]]:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(status, json=REPLY)

    http = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))
    return OllamaClient(model="m", num_ctx=2048, client=http), seen


def test_complete_returns_text_and_usage() -> None:
    client, seen = make_client()
    assert client.complete([Message(role="user", content="hi")]) == "hi there"
    usage = client.last_usage
    assert usage is not None
    assert (usage.prompt_tokens, usage.completion_tokens) == (21, 4)
    assert usage.tokens_per_second == 2.0
    assert seen[0]["options"]["num_ctx"] == 2048
    assert seen[0]["stream"] is False


def test_count_tokens_generates_one_token() -> None:
    client, seen = make_client()
    assert client.count_tokens([Message(role="user", content="hi")]) == 21
    assert seen[0]["options"]["num_predict"] == 1


def test_missing_model_gives_helpful_error() -> None:
    client, _ = make_client(status=404)
    with pytest.raises(RuntimeError, match="ollama pull m"):
        client.complete([Message(role="user", content="hi")])


def test_unreachable_server_gives_helpful_error() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    http = httpx.Client(base_url="http://test", transport=httpx.MockTransport(refuse))
    with pytest.raises(RuntimeError, match="ollama serve"):
        OllamaClient(client=http).complete([Message(role="user", content="hi")])
