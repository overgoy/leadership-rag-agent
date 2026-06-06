# Company Leadership RAG Agent

Given a company's domain, this system (1) finds the company's leadership from public
web sources and (2) exposes a chat interface to ask questions over the collected data
— *"Who's their CTO?"*, *"How many VPs do they have?"*, *"Who heads marketing?"*,
*"Where is their CEO based?"*

It uses **Hybrid Structured RAG with no vector database**: leaders are extracted into a
structured SQLite table for exact **Text-to-SQL** queries, with SQLite's native **FTS5**
full-text index over the `bio` column for keyword/semantic search. Every leader stores
its `source_url`, and the chat agent cites those sources when it answers.

---

## Architecture

```
            ┌─────────────────┐      ┌──────────────────┐      ┌──────────────────┐
 company →  │  scraper.py     │  →   │   database.py    │  ←   │     app.py       │  ← user
   URL      │  search +       │      │   SQLite schema, │      │  Streamlit chat, │   chat
            │  parallel LLM   │      │   FTS5, indexes  │      │  ReAct Text-to-  │
            │  extraction +   │      │   read-only exec │      │  SQL agent       │
            │  verification   │      │                  │      │                  │
            └─────────────────┘      └──────────────────┘      └──────────────────┘
                  write path                  data                  read-only path
```

Three layers, kept independent (separation of concerns):

| File | Responsibility |
|---|---|
| `src/scraper.py` | Tavily web search (domain-anchored) → **concurrent** LLM structured extraction (pydantic-validated) → skeptical employment verification → write to SQLite. |
| `src/database.py` | Schema, B-Tree indexes, FTS5 virtual table + sync triggers, WAL mode. Exposes a **read-only** `execute_sql` tool and a pruned `get_schema`. |
| `src/app.py` | Streamlit chat UI + ReAct Text-to-SQL agent. Reasons in `<thinking>` tags, calls the `execute_sql` tool, self-corrects on `<error>`, streams the answer, and cites `source_url`. |

---

## Engineering Strategy & Design Rationales

This section explains *why* the system is built the way it is — the trade-offs behind
each major decision.

### 1. Why a structured relational table (SQLite), not a vector database

Leadership analytics are **deterministic, structured queries**, not fuzzy semantic
recall. The canonical questions are *"how many VPs?"*, *"who is the CTO?"*,
*"list everyone in Engineering"*, *"where is the CEO based?"*. These are exact filters,
counts, and groupings over well-typed attributes (`role_category`, `department`,
`location`, `company`).

A relational table answers them **exactly and verifiably**:

```sql
SELECT count(*) FROM leadership WHERE role_category = 'VP' AND company = 'robinhood.com';
```

That query is correct, repeatable, and auditable. A pure vector database, by contrast,
would answer *"how many VPs?"* by embedding the question, retrieving the *k* nearest
bio chunks, and asking an LLM to count them — which is approximate, sensitive to `k`,
prone to silent under/over-counting, and impossible to verify. Vector search is the
right tool for *"find me people who sound like growth-stage operators"*; it is the
**wrong** tool for aggregates and precise attribute filters, which are exactly what
leadership analytics demand.

Concretely, SQLite gives us:
- **Exactness** — `COUNT`, `GROUP BY`, `WHERE role_category = 'C-Level'` return ground truth.
- **B-Tree indexes** on the heavily filtered columns (`role`, `department`,
  `role_category`, `company`) for fast lookups.
- **Zero infrastructure** — a single embedded file, no server, no embedding model, no
  similarity-tuning. Lean by design.
- **Auditability** — the agent's generated SQL is shown to the user, so every answer
  can be traced back to a concrete query and `source_url`.

### 2. Native SQLite FTS5 triggers for zero-cost, localized keyword search over bios

We still want free-text search over biographies (*"who works on payments
infrastructure?"*) — but without standing up a separate vector store or embedding
pipeline. SQLite's built-in **FTS5** extension provides this in-process and for free.

We define an external-content FTS5 virtual table over `(name, bio)` and keep it in
perfect sync with the base table using **`AFTER INSERT` / `AFTER UPDATE` / `AFTER DELETE`
triggers**:

```sql
CREATE VIRTUAL TABLE leadership_fts USING fts5(name, bio, content='leadership', content_rowid='id');

CREATE TRIGGER leadership_ai AFTER INSERT ON leadership BEGIN
    INSERT INTO leadership_fts(rowid, name, bio) VALUES (new.id, new.name, new.bio);
END;
-- (+ matching delete/update triggers)
```

Why this is the right call:
- **Zero extra cost / infrastructure** — FTS5 ships with the Python standard library's
  SQLite. No embedding API calls, no vector index to host, no extra dependency.
- **Localized & automatic** — the triggers mean the search index is *never* stale and
  the application code never has to remember to update it; inserting a leader is enough.
- **Good enough for bios** — BM25-ranked keyword search over short professional bios is
  fast and relevant, and it composes with the structured filters in a single SQL query
  (`JOIN leadership_fts ... WHERE leadership_fts MATCH '...'`).

### 3. Tavily as an isolated fetch sandbox for untrusted web HTML

Scraping arbitrary company websites means ingesting **untrusted, potentially hostile
content**: malformed HTML, tracking scripts, malicious markup, redirect chains, and
unbounded page weight. We deliberately **do not** fetch and parse raw DOM trees in our
own runtime (no in-process `requests` + `BeautifulSoup`, no headless browser executing
third-party JavaScript).

Instead, all fetching and HTML rendering happens **on Tavily's infrastructure**, which
acts as an isolation boundary. Tavily fetches the page and returns already-extracted
text (`include_raw_content=True`); our process only ever receives **plain text**, never
live HTML/JS/DOM. This shrinks our attack surface dramatically:
- No untrusted JavaScript ever executes in our environment.
- No HTML/XML parser in our process is exposed to adversarial markup (a classic source
  of parser exploits and billion-laughs/XXE-style attacks).
- No SSRF/redirect handling, TLS edge-cases, or unbounded downloads in our runtime.

We then defensively **truncate that text to 15,000 characters** before it reaches the
LLM, bounding both token cost and the blast radius of any pathological page.

### 4. Defense-in-depth database execution guards

LLM-generated SQL is, by definition, untrusted input. We assume the model *will*
eventually emit something dangerous (a `DROP`, a `DELETE`, an `UPDATE`, a write
disguised inside a CTE) and make that **impossible to execute**, with two independent
layers:

1. **Strict read-only connection (`mode=ro`)** — the agent reaches the database only
   through a URI connection opened read-only:
   ```python
   sqlite3.connect("file:company_data.db?mode=ro", uri=True, check_same_thread=False)
   ```
   At the SQLite engine level, *any* write or DDL fails with
   `attempt to write a readonly database`. This is the authoritative guarantee — even a
   cleverly obfuscated mutation cannot modify or destroy data. (`check_same_thread=False`
   also lets the Streamlit UI share connections across threads.)

2. **Application-level SELECT filtering** — before a query is even sent to SQLite, a
   guard rejects anything that isn't a read (`SELECT`/`WITH`), returning a polite
   `<error>` instead. This catches mutations early, gives the agent a clean,
   self-correctable error message, and means we are never *relying* on a single control.

The two layers are deliberately redundant: the regex guard is fast and user-friendly,
while the read-only connection is the hard backstop that holds even if the guard is ever
bypassed. Errors are returned (not raised) wrapped in `<error>` tags so the agent can
read the failure and fix its own SQL.

### 5. Scaling & observability architecture

The pipeline is built to scale to large domains and to be measurable in production.

**Bounded concurrency with backoff.** Page mining is the slow part of collection, and
each page is independent and network-bound (it waits on LLM calls). We fan those out
across a `ThreadPoolExecutor` so their latency overlaps — but the pool is deliberately
**bounded** (`MAX_WORKERS`, default 8). Unbounded concurrency would trip provider rate
limits; a bounded pool is the back-pressure mechanism. On top of that, every LLM call
sets `num_retries=3` with litellm's `retry_strategy="exponential_backoff_retry"`, so
transient errors and **HTTP 429** rate limits are retried with increasing delays rather
than failing the run. A single page that still fails is logged and skipped — it cannot
abort the whole collection.

**One atomic write, not many small ones.** Worker threads never touch the database.
They return their results to the main thread, which performs a single
`replace_company()` — a `DELETE` + bulk `INSERT` inside **one transaction** on one
connection. This avoids the classic SQLite *"database is locked"* contention you get
from many threads issuing small concurrent commits, and makes a re-scrape atomic: the
old rows are never visible-as-deleted without the new rows replacing them.

**Telemetry as a first-class table.** Every `make collect` run records a row in a
dedicated `system_metrics` table — `duration_seconds`, `pages_mined`,
`candidates_extracted`, `candidates_verified`, `tokens_used`, and
`estimated_cost_usd` (computed via `litellm.completion_cost`). Keeping metrics in their
own table (not mixed into `leadership`, and excluded from the agent's pruned schema)
means observability never pollutes the Text-to-SQL context. The Streamlit
**System Insights & Dashboard** view then reads this table through the same read-only
connection to surface KPI cards (total LLM cost, scraper success rate, total records)
and diagnostic `st.bar_chart`s (runtime by company, leadership distribution by
category) — turning the pipeline into something you can actually monitor and cost-track.

### Other notable decisions
- **Trustworthy provenance.** `company` and `source_url` are attached server-side (never
  taken from the model), so citations can't be hallucinated.
- **Precision over recall in sourcing.** Domain-anchored search avoids homonym
  companies; a deterministic board-title filter drops governance roles; a second-pass
  LLM verification drops customers/partners quoted in case studies.
- **Concurrency for scale.** Pages are mined in parallel with a `ThreadPoolExecutor`
  (see *Scalability* below).
- **FinOps.** Cost-effective "mini" models by default, bounded `max_tokens`, pruned
  schema, truncated web text and tool results.

---

## Scalability: concurrent page mining

For large domains the scraper mines pages **concurrently**. Each mined page is an
independent unit of work — LLM extraction followed by per-candidate employment
verification — and each is network-bound (it spends its time waiting on API calls), so a
`ThreadPoolExecutor` overlaps that latency without needing `asyncio`:

```python
with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(pages))) as pool:
    futures = {pool.submit(extract_leaders, page["content"], page["url"], company): idx
               for idx, page in enumerate(pages)}
    for future in as_completed(futures):
        per_page[futures[future]] = future.result()
```

Results are written into page-ordered slots, so the subsequent dedupe ("first occurrence
wins") stays **deterministic regardless of completion order**. A single failing page is
logged and skipped — it can't abort the whole run. The pool size is bounded
(`MAX_WORKERS`, default 8, override via `SCRAPER_MAX_WORKERS`) to respect API rate
limits, and each LLM call retries 429s with exponential backoff. The threads return
their results to the main thread, which commits them in **one atomic transaction**
(`replace_company`) to avoid SQLite lock contention.

Each run also writes a `system_metrics` row (duration, pages, candidates extracted vs.
verified, tokens, estimated USD cost), surfaced in the dashboard below.

---

## System Insights & Dashboard

The Streamlit app has two views, switchable from the sidebar radio:

- **💬 Chat Assistant** — the Text-to-SQL agent described above.
- **📊 System Insights & Dashboard** — observability over all collection runs:
  - **KPI cards** (`st.metric`): Total Infrastructure Cost (aggregated LLM spend),
    Scraper Success Rate (verified ÷ extracted), Total Records, and LLM Tokens.
  - **Diagnostic charts** (`st.bar_chart`): pipeline runtime by company, and leadership
    distribution by role category.
  - A **recent-runs table** of the raw `system_metrics` rows.

All dashboard data is read through the same read-only connection as the agent.

---

## Deployment (optional)

Local execution via `make chat` is the primary path. For cloud hosting, a `Dockerfile`
is included:

```bash
docker build -t leadership-agent .
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=... -e TAVILY_API_KEY=... \
  -v "$(pwd)/data:/app/data" \
  leadership-agent
```

It runs Streamlit headless on port 8501 with a health-check against
`/_stcore/health`. Mount `data/` as a volume to provide (or persist) the SQLite
database; secrets are passed as environment variables, never baked into the image.

---

## Prerequisites
- Python 3.11+ (the standard library must include the SQLite **FTS5** extension — it does on official CPython builds).
- An **OpenAI** API key and a **Tavily** API key.

---

## Setup (clone-and-go)

```bash
# 1. Install (creates a venv and installs dependencies)
make install

# 2. Configure API keys
cp .env.example .env
#   then edit .env and fill in OPENAI_API_KEY and TAVILY_API_KEY

# 3. Collect leadership data for a company (writes data/company_data.db)
make collect URL=https://meetcampfire.com/
#   or the other test company:
make collect URL=https://robinhood.com/

# 4. Launch the chat interface  →  http://localhost:8501
make chat
```

> A pre-collected fixture for `meetcampfire.com` is included in the repo as a data
> sample (see **Data fixtures** below). You can skip step 3 and go straight to `make chat`
> if you only want to explore that company.

### Make targets

| Target | What it does |
|---|---|
| `make install` | Create `venv/` and install `requirements.txt`. |
| `make collect URL=<company_url>` | Scrape + extract + store leadership for one company. Idempotent (re-running replaces that company's rows). Defaults to `meetcampfire.com`. |
| `make chat` | Launch the Streamlit chat app (headless) on port 8501. |
| `make test` | Run the offline test suite (`pytest`). |
| `make eval` | Run the live agent eval — the task's example questions through the real agent, checked against the DB (needs `OPENAI_API_KEY` + collected data). |

---

## Using the chat

Ask natural-language questions; the agent translates them to SQL, runs them read-only,
and answers with citations. Examples:

- *Who are the C-level executives at meetcampfire.com?*
- *How many people are in each role category?*
- *Who heads engineering?*
- *List the co-founders and where they're based.*

Each answer has a **🧠 Reasoning & SQL** expander showing the agent's chain of thought,
the exact SQL it ran, and the rows returned — for full transparency. Off-topic questions
are politely declined.

---

## Data model

A single `leadership` table:

| column | notes |
|---|---|
| `company` | company domain this leader belongs to (e.g. `meetcampfire.com`) |
| `name` | required |
| `role` | full title, e.g. "Chief Technology Officer" |
| `role_category` | one of `C-Level`, `VP`, `Head` (target scope) |
| `department` | e.g. "Engineering", "Marketing" |
| `location` | where the person is based |
| `bio` | free text, indexed by FTS5 |
| `linkedin_url` | if found |
| `source_url` | provenance — where the data was found |
| `is_active` / `valid_from` / `valid_to` | SCD-2 history: `is_active = 1`, `valid_to = NULL` is current; re-collects close the old window |

B-Tree indexes on `role`, `department`, `role_category`, `company`, `is_active`; an FTS5
virtual table `leadership_fts` over `(name, bio)` kept in sync via triggers.

The table is kept **flat** so Text-to-SQL stays join-free. Two supplementary dimensions
normalize company- and source-level facts (they are not queried by the agent):

- `companies` — `domain` (unique) → `display_name`, `hq_location`
- `sources` — deduplicated provenance `url` + `fetched_at` (data freshness)

`department` and `location` are **canonicalized on write** (whitespace + a small
synonym/alias map, e.g. `eng`→`Engineering`, `SF`→`San Francisco, CA`) so re-collects and
`GROUP BY`/filters don't fragment on case or synonym variants.

---

## Data fixtures

A pre-collected dataset is **committed** at `data/company_data.db` so the project is
clone-and-go — you can `make chat` and query immediately without any API keys. It
contains both example companies (`meetcampfire.com` and `robinhood.com`), with
enriched bios (used by FTS5) and locations where the source pages stated them, and
correctly excludes board members and customers quoted in case studies.

Regenerate or extend it anytime:

```bash
make collect DOMAIN=meetcampfire.com
make collect DOMAIN=robinhood.com
```

(The DB's transient `-wal`/`-shm` sidecars stay git-ignored.) Notes: `location`
coverage depends on what each page states, and `linkedin_url` is typically empty by
design — we ingest Tavily's extracted *text*, not raw HTML anchors (see the security
rationale above), so profile hrefs usually aren't present to bind.

---

## Configuration

Environment variables (see `.env.example`):

| var | required | default | used by |
|---|---|---|---|
| `OPENAI_API_KEY` | yes | — | extraction, verification, agent |
| `TAVILY_API_KEY` | yes | — | web search |
| `EXTRACTION_MODEL` | no | `gpt-4o-mini` | `src/scraper.py` |
| `AGENT_MODEL` | no | `gpt-4o-mini` | `src/app.py` |
| `SCRAPER_MAX_WORKERS` | no | `8` | `src/scraper.py` (concurrency) |

---

## Development

```bash
venv/bin/ruff check . --fix   # lint
venv/bin/ruff format .        # format
make test                     # offline unit tests (or: venv/bin/pytest)
make eval                     # live agent eval against the real LLM + DB
```

The test suite is offline and deterministic — it exercises the SQL layer (schema,
FTS5, read-only enforcement) and the pure text/guard helpers, and never calls a
real LLM or the network. `make eval` (`eval_agent.py`) complements it by running
the task's example questions through the real agent and checking each answer
against ground truth read from the database — kept out of `tests/` precisely so
the unit suite stays offline.

## Project layout

```
.
├── data/                # SQLite database (generated; .gitkeep tracked)
├── src/
│   ├── scraper.py       # web search + parallel LLM extraction + verification
│   ├── database.py      # SQLite schema, FTS5, read-only execution tools
│   └── app.py           # Streamlit chat UI + Text-to-SQL agent
├── tests/               # pytest: database, scraper, and agent helpers
├── conftest.py          # test fixtures (temp DB) + src/ import path
├── eval_agent.py        # live agent eval (real LLM) — `make eval`
├── .streamlit/          # Streamlit config (telemetry off)
├── requirements.txt
├── .env.example         # template for API keys
├── Dockerfile           # optional: containerized deploy of the Streamlit app
├── pyproject.toml       # ruff lint/format configuration
├── session.json         # Claude Code build export (tracked deliverable)
└── Makefile
```