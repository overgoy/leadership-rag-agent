# Project Map — Company Leadership RAG Agent

A context map for AI agents and new contributors. For setup and rationale see
`README.md`; for the strict build rules see `.claudecode.md`.

## What this is

An end-to-end **Hybrid Structured RAG** system (no vector DB). Given a company
domain it (1) collects leadership data from the public web into SQLite and (2)
serves a Streamlit chat agent that answers questions via **Text-to-SQL**, plus a
telemetry dashboard. Leaders are stored in a structured table (exact filters,
counts) with an **FTS5** index over bios (keyword search).

## Architecture & data flow

```
company URL ─▶ scraper.py ─▶ SQLite (database.py) ◀─ app.py ◀─ user
              (write path)      data layer            (read-only path)
```

- **Write path** (`scraper.py`): Tavily domain-anchored search → concurrent LLM
  extraction (pydantic) → employment verification → atomic soft-delete write +
  metrics.
- **Read path** (`app.py`): ReAct agent emits SQL → `execute_sql` (read-only) →
  cited answer; dashboard reads `system_metrics`.
- The three layers are decoupled: the app reaches data **only** through
  `database.execute_sql` / `database.get_schema`.

## Files

| Path | Role |
|---|---|
| `src/database.py` | SQLite schema, FTS5 + triggers, indexes, WAL. Functions: `init_db`, `insert_leaders`, `replace_company` (atomic soft-delete + insert), `clear_company`, `insert_metrics`, `get_schema` (pruned, for the LLM), `execute_sql` (read-only). |
| `src/scraper.py` | `search_company` (Tavily), `extract_leaders` (LLM + filters + verify, returns `(leaders, stats)`), `verify_employment`, `collect` (concurrent mining, metrics). CLI: `python -m src.scraper <url>`. |
| `src/app.py` | Streamlit app. `render_chat` (ReAct Text-to-SQL, streamed, cited), `render_dashboard` (KPIs + charts), `run_agent`, `_run_tool`, guards (`_is_select`), `_visible_answer`/`_ThinkingFilter`. |
| `tests/` + `conftest.py` | 49 offline pytest tests (no LLM/network). `temp_db` fixture. |
| `Makefile` | `install`, `collect URL=…`, `chat`, `test`. |
| `Dockerfile` / `.dockerignore` | Optional containerized deploy (port 8501). |
| `pyproject.toml` | ruff lint/format config. |
| `.streamlit/config.toml` | Streamlit config (telemetry off). |
| `session.json` | Claude Code build-session export (deliverable). |

## Database schema

**`leadership`** — one row per leader (current and historical):

| column | type | notes |
|---|---|---|
| `id` | INTEGER PK | autoincrement |
| `company` | TEXT | domain, e.g. `meetcampfire.com` |
| `name` | TEXT NOT NULL | |
| `role` | TEXT | full title |
| `role_category` | TEXT | `C-Level` \| `VP` \| `Head` |
| `department` | TEXT | |
| `location` | TEXT | |
| `bio` | TEXT | indexed by FTS5 |
| `linkedin_url` | TEXT | |
| `source_url` | TEXT | provenance (attached server-side) |
| `created_at` | TIMESTAMP | default `CURRENT_TIMESTAMP` |
| `is_active` | INTEGER | `1` = current, `0` = soft-deleted/historical |
| `valid_from` | TIMESTAMP | when the record became current |

- **Indexes**: B-Tree on `role`, `department`, `role_category`, `company`.
- **FTS5**: `leadership_fts(name, bio)` external-content over `leadership`, kept in
  sync by `AFTER INSERT/UPDATE/DELETE` triggers.
- **History**: `replace_company` sets prior rows `is_active = 0` and inserts new
  ones as `is_active = 1`, in one transaction. **Always filter `WHERE is_active = 1`**
  for present-day questions; include `is_active = 0` only for history.

**`system_metrics`** — one row per `make collect` run:
`id`, `company`, `created_at`, `duration_seconds`, `pages_mined`,
`candidates_extracted`, `candidates_verified`, `tokens_used`,
`estimated_cost_usd`. (Excluded from `get_schema`, so it never enters the agent's
Text-to-SQL context; the dashboard queries it directly.)

## Key conventions

- **Security**: the agent's DB connection is read-only (`mode=ro`); an
  application-level `SELECT`-only guard sits on top (defense in depth).
- **FinOps**: cheap "mini" models by default, bounded `max_tokens`, pruned schema,
  truncated web text (15k chars) and tool results.
- **Resilience**: pydantic Optional fields, `num_retries=3` with exponential
  backoff on 429s, bounded `ThreadPoolExecutor`, per-page failures are skipped.
- **Provenance**: `company`/`source_url` set server-side, never from the model;
  the agent cites `source_url`.
- **Tooling**: `ruff check . --fix` + `ruff format .` before commits; tests are
  offline and must stay green (`make test`).