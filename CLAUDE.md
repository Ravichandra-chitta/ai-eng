# Project rules for AI-assisted development

This repo is a learning and portfolio project for AI engineering. Work in small steps.

## Workflow (applies to Claude Code, Cursor, any AI tool)
1. **Spec first.** Before coding, write 3-6 lines in the PR/commit or `docs/` describing the behavior.
2. **Tests first.** Write or update a failing test, then the code that makes it pass.
3. **Review every diff.** Read all AI-generated changes before committing. If you cannot explain a line, rewrite or remove it.
4. **Log the misses.** Add an entry to `ai-dev-log.md` whenever the AI got something wrong, invented an API, or needed a correction.

## Commands
- Install: `uv sync`
- Test: `uv run pytest`
- Lint/format: `uv run ruff check .` and `uv run ruff format .`
- Types: `uv run mypy`
- Run all checks before every commit.

## Code rules
- Python 3.12+, full type hints (mypy strict), Pydantic for data models.
- No network or API keys in unit tests; use fakes (see `ai_eng.llm.FakeClient`).
- Never commit `.env` or any secret. Personal accounts and keys only, with spend caps.
- Use public data only. No employer code, data or internal designs.
