"""Anthropic implementation of the provider-agnostic `LLMClient` protocol."""

from collections.abc import Sequence
from dataclasses import dataclass

import anthropic
from anthropic.types import MessageParam

from ai_eng.llm import Message

DEFAULT_MODEL = "claude-haiku-5-5"

# USD per million tokens for Claude Haiku 5.5 with prompts <= 100K tokens.
# Check https://platform.claude.com/docs/en/about-claude/pricing before relying on these.
INPUT_USD_PER_MTOK = 0.10
OUTPUT_USD_PER_MTOK = 0.50


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int

    @property
    def cost_usd(self) -> float:
        return (
            self.input_tokens * INPUT_USD_PER_MTOK + self.output_tokens * OUTPUT_USD_PER_MTOK
        ) / 1_000_000


def split_messages(messages: Sequence[Message]) -> tuple[str | None, list[MessageParam]]:
    """The Messages API takes `system` separately from user/assistant turns."""
    system_parts: list[str] = []
    turns: list[MessageParam] = []
    for m in messages:
        if m.role == "system":
            system_parts.append(m.content)
        elif m.role == "user":
            turns.append({"role": "user", "content": m.content})
        else:
            turns.append({"role": "assistant", "content": m.content})
    return ("\n\n".join(system_parts) or None), turns


class AnthropicClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 1024,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        # With no argument the SDK reads the ANTHROPIC_API_KEY environment variable.
        self._client = client or anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.last_usage: Usage | None = None

    def complete(self, messages: Sequence[Message]) -> str:
        system, turns = split_messages(messages)
        if system is None:
            response = self._client.messages.create(
                model=self.model, max_tokens=self.max_tokens, messages=turns
            )
        else:
            response = self._client.messages.create(
                model=self.model, max_tokens=self.max_tokens, system=system, messages=turns
            )
        self.last_usage = Usage(response.usage.input_tokens, response.usage.output_tokens)
        return "".join(block.text for block in response.content if block.type == "text")

    def count_tokens(self, messages: Sequence[Message]) -> int:
        """Free endpoint: how many input tokens would these messages use?"""
        system, turns = split_messages(messages)
        if system is None:
            result = self._client.messages.count_tokens(model=self.model, messages=turns)
        else:
            result = self._client.messages.count_tokens(
                model=self.model, system=system, messages=turns
            )
        return result.input_tokens
