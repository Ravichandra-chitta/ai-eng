from types import SimpleNamespace
from typing import Any

from ai_eng.anthropic_client import AnthropicClient, Usage, split_messages
from ai_eng.llm import Message


class FakeMessages:
    def __init__(self) -> None:
        self.create_kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.create_kwargs = kwargs
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="hello")],
            usage=SimpleNamespace(input_tokens=12, output_tokens=3),
        )

    def count_tokens(self, **kwargs: Any) -> Any:
        return SimpleNamespace(input_tokens=7)


def make_client() -> tuple[AnthropicClient, FakeMessages]:
    messages = FakeMessages()
    fake_sdk: Any = SimpleNamespace(messages=messages)
    return AnthropicClient(client=fake_sdk), messages


def test_split_messages_moves_system_out() -> None:
    system, turns = split_messages(
        [
            Message(role="system", content="be brief"),
            Message(role="user", content="hi"),
        ]
    )
    assert system == "be brief"
    assert turns == [{"role": "user", "content": "hi"}]


def test_split_messages_without_system() -> None:
    system, turns = split_messages([Message(role="user", content="hi")])
    assert system is None
    assert len(turns) == 1


def test_complete_returns_text_and_records_usage() -> None:
    client, messages = make_client()
    reply = client.complete(
        [Message(role="system", content="s"), Message(role="user", content="q")]
    )
    assert reply == "hello"
    assert messages.create_kwargs["system"] == "s"
    assert client.last_usage == Usage(12, 3)


def test_count_tokens() -> None:
    client, _ = make_client()
    assert client.count_tokens([Message(role="user", content="hi")]) == 7


def test_cost_usd() -> None:
    # 1M input tokens at $0.10 plus 1M output tokens at $0.50
    assert Usage(1_000_000, 1_000_000).cost_usd == 0.6
