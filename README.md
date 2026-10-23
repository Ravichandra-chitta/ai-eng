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

## Layout

- `src/ai_eng/`: library code
- `tests/`: pytest tests (no network, use fakes)
- `.github/workflows/ci.yml`: lint, types and tests on every push and PR
- `CLAUDE.md`: rules for AI-assisted development
- `ai-dev-log.md`: log of what AI tools got wrong and what I learned
