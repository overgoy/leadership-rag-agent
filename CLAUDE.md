# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

The repo uses a `venv/` and a Makefile; there is no global install. Use `venv/bin/...` directly.

```bash
make install                       # create venv + install requirements.txt
make collect DOMAIN=robinhood.com  # scrape one company into data/company_data.db (or URL=...)
make chat                          # launch the Streamlit app (headless, port 8501)
make test                          # offline pytest suite (no LLM, no network)
make eval                          # live agent eval — real LLM against the collected DB

venv/bin/ruff check . --fix        # lint  (run BOTH before any commit — see Working rules)
venv/bin/ruff format .             # format
venv/bin/pytest tests/test_app.py::test_is_select_allows_reads   # a single test
```

`make collect`, `make chat`, and `make eval` need `OPENAI_API_KEY` and `TAVILY_API_KEY` in `.env` (template: `.env.example`). `make test` needs neither.

## Architecture

A **Hybrid Structured RAG** system (no vector DB): leadership is scraped into a structured SQLite table for exact Text-to-SQL, with FTS5 over bios for keyword search. Three layers are deliberately decoupled — the app reaches data **only** through `database.execute_sql` / `get_schema`, never the scraper:

- `src/scraper.py` (write path) — Tavily domain-anchored search → concurrent (`ThreadPoolExecutor`) LLM extraction → skeptical employment verification → atomic write. `extract_leaders()` returns `(leaders, stats)`; `collect()` aggregates stats, runs the HQ-location backfill, writes once, and logs a `system_metrics` row.
- `src/database.py` (data layer) — schema, indexes, FTS5 + sync triggers, WAL, migration. Read-only vs read-write connections.
- `src/app.py` (read path) — Streamlit UI + ReAct Text-to-SQL agent, plus a System Insights dashboard. Two sidebar views: chat and dashboard.

Read `.claudecode.md` (the authoritative spec, referenced as §1–§8 throughout the code) before substantial changes. The full DB schema lives in `src/database.py` (`_SCHEMA`).

### Invariants that span multiple files — do not break these

- **Read-only agent + defense in depth.** The agent's DB access is read-only (`file:...?mode=ro` in `database._connect`) AND guarded by an app-level SELECT-only check (`app._is_select`). Both layers must stay; never give the agent a write path.
- **Soft-delete history.** `replace_company()` does NOT hard-delete: it flags prior rows `is_active = 0` and inserts new rows as `is_active = 1`, in one transaction. **Always filter `WHERE is_active = 1` for current data** (the agent prompt and dashboard already do). `clear_company()` still hard-deletes and is kept only for tests.
- **FTS5 is trigger-maintained.** `leadership_fts` is external-content over `leadership`, synced by AFTER INSERT/UPDATE/DELETE triggers — don't write to it directly.
- **Provenance is server-side.** `company` and `source_url` are attached in `extract_leaders` from the crawl context, never taken from the model. Keep it that way so citations can't be hallucinated.
- **`system_metrics` is excluded from `get_schema()`** so it never enters the agent's Text-to-SQL context; the dashboard queries it directly.

### Agent (`src/app.py`) specifics

- `run_agent()` is a **generator** that yields only the final answer's tokens for `st.write_stream`; tool calls and `<thinking>` go into a `steps` list rendered in an expander. `_ThinkingFilter` strips `<thinking>` live, even across split tokens.
- The system prompt (`_build_system_prompt`) encodes the data quirks the model must respect: `company` is a **domain** (`robinhood.com`), `role` is **free text**, titles need **acronym+full-form** matching combined with `role_category` (e.g. `role_category='C-Level' AND (role LIKE '%ceo%' OR role LIKE '%chief executive officer%')`). SQLite `LIKE` is case-insensitive for ASCII — keep generated SQL lean (no `LOWER()`).
- **Multi-company questions** ("Who are the CEOs?") query across all companies and answer with a per-company breakdown — they do NOT short-circuit for clarification.
- **"Did you mean?"**: when a tool query returns 0 rows, `run_agent` appends the active-company list to the tool result so the agent suggests corrections.
- Off-topic requests must refuse on the first step without emitting SQL.

### Data quirks

- `company` is stored as a bare domain. `role`/`name` matching uses `LIKE` fragments, not `=`.
- `location` coverage comes from page text plus the HQ backfill (`resolve_hq_location`: dedicated Tavily HQ search, then falls back to the modal extracted location). `linkedin_url` is usually empty by design — we ingest Tavily's extracted *text*, not raw HTML anchors.

## Working rules (from `.claudecode.md` §8)

- Run `ruff check . --fix` and `ruff format .` before proposing any commit.
- Keep `make test` offline and deterministic — **no mocked LLM completions**. Live/agent checks belong in `eval_agent.py` (`make eval`), not `tests/`.
- Lean and maintainable; no over-engineering. Default models are cost-effective "mini" tiers with bounded `max_tokens`.

## Fixtures & the committed database

`data/company_data.db` is a **committed fixture** (clone-and-go, no keys needed) — `*.db` is git-ignored except this file (via a `!` negation in `.gitignore`); the `-wal`/`-shm` sidecars stay ignored. After (re)collecting, run a WAL checkpoint before committing so the `.db` is standalone:

```python
import sqlite3; from src import database as d
sqlite3.connect(d.DB_PATH).execute("PRAGMA wal_checkpoint(TRUNCATE);")
```

`session.json` is a committed deliverable (a redacted export of the build session). `make collect` is idempotent per company; re-collecting is non-deterministic (LLM + live search), so leader counts vary slightly between runs.