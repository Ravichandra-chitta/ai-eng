"""Provider-agnostic LLM interface.

Seed for the Phase 1 provider abstraction: hosted and local models will sit behind
the same `LLMClient` protocol. `FakeClient` lets tests run with no network or API key.
"""

from collections.abc import Sequence
from typing import Literal, Protocol

from pydantic import BaseModel

Role = Literal["system", "user", "assistant"]


class Message(BaseModel):
    role: Role
    content: str


class LLMClient(Protocol):
    def complete(self, messages: Sequence[Message]) -> str:
        """Return the assistant's reply for the given conversation."""
        ...


class FakeClient:
    """Deterministic stand-in used in tests."""

    def __init__(self, reply: str = "ok") -> None:
        self.reply = reply
        self.calls: list[list[Message]] = []

    def complete(self, messages: Sequence[Message]) -> str:
        self.calls.append(list(messages))
        return self.reply


def ask(client: LLMClient, question: str, system: str | None = None) -> str:
    """Send one question (with an optional system prompt) and return the reply."""
    messages: list[Message] = []
    if system:
        messages.append(Message(role="system", content=system))
    messages.append(Message(role="user", content=question))
    return client.complete(messages)
