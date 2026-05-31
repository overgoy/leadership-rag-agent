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
limits.

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
| `make test` | Run the test suite (`pytest`). |

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

B-Tree indexes on `role`, `department`, `role_category`, `company`; an FTS5 virtual table
`leadership_fts` over `(name, bio)` kept in sync via triggers.

---

## Data fixtures

`make collect URL=https://meetcampfire.com/` produces a clean fixture of Campfire's
leadership (founders + heads), correctly excluding board members and customers quoted in
case studies. The SQLite database itself (`data/company_data.db`) is git-ignored as a
generated artifact; regenerate it with a single `make collect`.

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
make test                     # run the tests (or: venv/bin/pytest)
```

The test suite is offline and deterministic — it exercises the SQL layer (schema,
FTS5, read-only enforcement) and the pure text/guard helpers, and never calls a
real LLM or the network.

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
├── .streamlit/          # Streamlit config (telemetry off)
├── requirements.txt
├── .env.example         # template for API keys
├── session.json         # how this was built with Claude Code (deliverable)
└── Makefile
```