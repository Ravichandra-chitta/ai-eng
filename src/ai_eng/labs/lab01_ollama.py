"""Lab 1 (local): tokens and context with Ollama models.

Run:  uv run python -m ai_eng.labs.lab01_ollama phi4-mini qwen3.5:4b
Needs Ollama running and each model pulled. Free, but slow on CPU: be patient.
Run one model at a time on 8 GB RAM (see `ollama ps`).
"""

import sys

from ai_eng.labs.lab01_tokens import SAMPLES, tokens_per_char
from ai_eng.llm import Message
from ai_eng.ollama_client import OllamaClient

SECRET = "The secret code word is PINEAPPLE."
FILLER = "The committee reviewed the quarterly logistics report in detail. "


def token_table(client: OllamaClient) -> None:
    print(f"\n== {client.model}: tokens per sample ==")
    print(f"{'sample':<10}{'chars':>7}{'tokens':>8}{'tok/char':>10}")
    for name, text in SAMPLES.items():
        n = client.count_tokens([Message(role="user", content=text)])
        print(f"{name:<10}{len(text):>7}{n:>8}{tokens_per_char(n, text):>10.2f}")


def overflow_demo(client: OllamaClient, repeats: int = 400) -> None:
    """Put a secret first, then far more filler than num_ctx can hold, then ask for it."""
    prompt = (
        SECRET + "\n" + FILLER * repeats + "\nWhat is the secret code word? Answer in one word."
    )
    reply = client.complete([Message(role="user", content=prompt)])
    usage = client.last_usage
    print(f"\n== {client.model}: context overflow (num_ctx={client.num_ctx}) ==")
    print(f"Prompt characters sent: {len(prompt)}")
    if usage:
        print(f"prompt_eval_count reported: {usage.prompt_tokens}")
        print(f"speed: {usage.tokens_per_second:.1f} tok/s, total {usage.total_seconds:.1f}s")
    print(f"Reply: {reply.strip()[:200]}")
    print("Did it know the code word? Compare prompt_eval_count with num_ctx.")


def main(models: list[str]) -> None:
    for model in models or ["phi4-mini"]:
        client = OllamaClient(model=model, num_ctx=2048)
        token_table(client)
        overflow_demo(client)


if __name__ == "__main__":
    main(sys.argv[1:])
