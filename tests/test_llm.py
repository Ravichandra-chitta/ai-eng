import pytest
from pydantic import ValidationError

from ai_eng.llm import FakeClient, Message, ask


def test_ask_returns_reply() -> None:
    assert ask(FakeClient("hello"), "hi") == "hello"


def test_ask_includes_system_prompt_first() -> None:
    client = FakeClient()
    ask(client, "question", system="be brief")
    assert [m.role for m in client.calls[0]] == ["system", "user"]


def test_ask_without_system_prompt() -> None:
    client = FakeClient()
    ask(client, "question")
    assert [m.role for m in client.calls[0]] == ["user"]


def test_message_rejects_unknown_role() -> None:
    with pytest.raises(ValidationError):
        Message(role="robot", content="x")
