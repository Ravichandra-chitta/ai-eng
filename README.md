# ai-eng

Hands-on AI engineering: LLM apps, RAG, agents, evals and cloud deployment (Python-first).

## Quick start

```bash
uv sync            # create .venv and install everything
uv run pytest      # run tests
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

Copy `.env.example` to `.env` and add personal API keys (never commit `.env`).

## Debugging `lab01_tokens`

- Run `uv sync` and select the workspace `.venv` interpreter in VS Code.
- Open Run and Debug and start **Debug lab01_tokens (uv)**.
- The launch profile runs the lab module from the repository root and loads ignored `.env`.
- The current lab uses Anthropic's token-count endpoint; its completion call remains commented out.

## Layout

- `src/ai_eng/`: library code
- `tests/`: pytest tests (no network, use fakes)
- `.github/workflows/ci.yml`: lint, types and tests on every push and PR
- `CLAUDE.md`: rules for AI-assisted development
- `ai-dev-log.md`: log of what AI tools got wrong and what I learned
