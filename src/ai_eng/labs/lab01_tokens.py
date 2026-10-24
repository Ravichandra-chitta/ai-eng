"""Lab 1: tokens and context.

Run:  uv run python -m ai_eng.labs.lab01_tokens
Needs ANTHROPIC_API_KEY. Part 1 uses the free token-counting endpoint;
part 2 makes ONE small paid call (a fraction of a cent) to show real usage.
"""

from ai_eng.anthropic_client import AnthropicClient
from ai_eng.llm import Message

SAMPLES: dict[str, str] = {
    "english": "The quick brown fox jumps over the lazy dog.",
    "numbers": "1234567890 3.14159265358979 2026-10-24T11:53:00+05:30",
    "python": "def add(a: int, b: int) -> int:\n    return a + b\n",
    "json": '{"user_id": 42, "active": true, "tags": ["a", "b"]}',
    "hindi": "मुझे कृत्रिम बुद्धिमत्ता इंजीनियर बनना है।",
    "telugu": "నేను AI ఇంజనీర్ కావాలనుకుంటున్నాను.",
}


def tokens_per_char(tokens: int, text: str) -> float:
    return tokens / len(text)


def main() -> None:
    client = AnthropicClient()

    print(f"{'sample':<10}{'chars':>7}{'words':>7}{'tokens':>8}{'tok/char':>10}")
    for name, text in SAMPLES.items():
        n = client.count_tokens([Message(role="user", content=text)])
        # The count includes a small fixed overhead for message framing.
        print(
            f"{name:<10}{len(text):>7}{len(text.split()):>7}{n:>8}{tokens_per_char(n, text):>10.2f}"
        )

    baseline = client.count_tokens([Message(role="user", content="hi")])
    print(f"\nFraming overhead (one-word message): {baseline} tokens")

    reply = client.complete([Message(role="user", content="Explain a token in one sentence.")])
    usage = client.last_usage
    print(f"\nReply: {reply}")
    if usage:
        print(f"Usage: {usage.input_tokens} in / {usage.output_tokens} out = ${usage.cost_usd:.6f}")


if __name__ == "__main__":
    main()
