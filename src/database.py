"""SQLite schema initialization, FTS5 setup, indexing, and execution tools.

This is the database execution layer. It is intentionally independent of the
scraper (data collection) and the Streamlit app (UI) — see .claudecode.md §2
"Separation of Concerns".

Design (see .claudecode.md):
- §2 Hybrid Structured RAG: a structured ``leadership`` table for exact
  Text-to-SQL matching, plus an FTS5 virtual table over the ``bio`` column for
  semantic/context queries. No vector DB.
- §2 Indexing: B-Tree indexes on the heavily filtered ``role`` and
  ``department`` columns.
- §2 Provenance: every row stores its ``source_url``.
- §3 Stability: ``PRAGMA journal_mode=WAL`` for concurrent read/write.
- §4 Security: the agent connects in read-only mode via a URI
  (``file:...?mode=ro``) with ``check_same_thread=False`` for the Streamlit UI.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Mapping, Sequence

# data/company_data.db at the project root (see .claudecode.md §7).
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "company_data.db"

# Columns we accept on insert. Only ``name`` is required; everything else is
# Optional to mirror the scraper's pydantic model and avoid losing partial data.
_LEADER_COLUMNS = (
    "company",
    "name",
    "role",
    "role_category",
    "department",
    "location",
    "bio",
    "linkedin_url",
    "source_url",
)

# Columns written per collection run into ``system_metrics`` (observability).
_METRIC_COLUMNS = (
    "company",
    "duration_seconds",
    "pages_mined",
    "candidates_extracted",
    "candidates_verified",
    "tokens_used",
    "estimated_cost_usd",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS leadership (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    company       TEXT,                       -- company domain/name this leader belongs to
    name          TEXT NOT NULL,
    role          TEXT,                       -- full title, e.g. "Chief Technology Officer"
    role_category TEXT,                       -- one of: 'C-Level', 'VP', 'Head'
    department    TEXT,                       -- e.g. "Marketing", "Engineering"
    location      TEXT,                       -- where the person is based
    bio           TEXT,                       -- free-text profile, indexed by FTS5
    linkedin_url  TEXT,
    source_url    TEXT,                       -- provenance: where this was found
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- data vitality / freshness
    is_active     INTEGER  DEFAULT 1,         -- 1 = current, 0 = soft-deleted (historical)
    valid_from    TIMESTAMP DEFAULT CURRENT_TIMESTAMP  -- when this record became current
);

-- B-Tree indexes on the heavily filtered columns (.claudecode.md §2).
CREATE INDEX IF NOT EXISTS idx_leadership_role        ON leadership(role);
CREATE INDEX IF NOT EXISTS idx_leadership_department  ON leadership(department);
CREATE INDEX IF NOT EXISTS idx_leadership_category    ON leadership(role_category);
CREATE INDEX IF NOT EXISTS idx_leadership_company     ON leadership(company);

-- FTS5 over the bio (plus name) for semantic/context queries. External-content
-- table keeps it in sync with ``leadership`` via the triggers below.
CREATE VIRTUAL TABLE IF NOT EXISTS leadership_fts USING fts5(
    name,
    bio,
    content='leadership',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS leadership_ai AFTER INSERT ON leadership BEGIN
    INSERT INTO leadership_fts(rowid, name, bio) VALUES (new.id, new.name, new.bio);
END;

CREATE TRIGGER IF NOT EXISTS leadership_ad AFTER DELETE ON leadership BEGIN
    INSERT INTO leadership_fts(leadership_fts, rowid, name, bio)
        VALUES ('delete', old.id, old.name, old.bio);
END;

CREATE TRIGGER IF NOT EXISTS leadership_au AFTER UPDATE ON leadership BEGIN
    INSERT INTO leadership_fts(leadership_fts, rowid, name, bio)
        VALUES ('delete', old.id, old.name, old.bio);
    INSERT INTO leadership_fts(rowid, name, bio) VALUES (new.id, new.name, new.bio);
END;

-- Observability: one row per `make collect` run, capturing performance and
-- FinOps data (.claudecode.md §3/§5). Kept separate from the agent's schema so
-- it never pollutes the Text-to-SQL context (the dashboard queries it directly).
CREATE TABLE IF NOT EXISTS system_metrics (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    company              TEXT,
    created_at           TEXT DEFAULT (datetime('now')),  -- UTC run timestamp
    duration_seconds     REAL,     -- wall-clock for the collection run
    pages_mined          INTEGER,  -- pages returned by search and mined
    candidates_extracted INTEGER,  -- raw leaders proposed by the extraction model
    candidates_verified  INTEGER,  -- survivors of scope + board + employment checks
    tokens_used          INTEGER,  -- total LLM tokens (extraction + verification)
    estimated_cost_usd   REAL      -- litellm cost estimate for those tokens
);

CREATE INDEX IF NOT EXISTS idx_metrics_company ON system_metrics(company);
"""


def _connect(read_only: bool = False) -> sqlite3.Connection:
    """Open a connection via a URI.

    ``read_only=True`` opens with ``mode=ro`` so LLM-generated SQL can never
    write or run destructive statements (.claudecode.md §4). All connections set
    ``check_same_thread=False`` so the Streamlit UI can share them across
    threads (.claudecode.md §4).
    """
    if read_only:
        uri = f"file:{DB_PATH}?mode=ro"
    else:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        uri = f"file:{DB_PATH}?mode=rwc"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the schema, FTS5 table, triggers, and indexes; enable WAL."""
    conn = _connect(read_only=False)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")  # §3 concurrent read/write
        conn.executescript(_SCHEMA)
        conn.commit()
    finally:
        conn.close()


def insert_leaders(leaders: Sequence[Mapping[str, Any]]) -> int:
    """Insert leader records. Accepts plain mappings to stay decoupled from the
    scraper's pydantic models. Unknown keys are ignored; missing keys become
    NULL. Returns the number of rows inserted.
    """
    if not leaders:
        return 0

    columns = ", ".join(_LEADER_COLUMNS)
    placeholders = ", ".join(f":{c}" for c in _LEADER_COLUMNS)
    sql = f"INSERT INTO leadership ({columns}) VALUES ({placeholders})"
    rows = [{c: dict(leader).get(c) for c in _LEADER_COLUMNS} for leader in leaders]

    conn = _connect(read_only=False)
    try:
        conn.executemany(sql, rows)
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def clear_company(company: str) -> int:
    """Delete existing rows for a company so a re-scrape doesn't duplicate.
    Returns the number of rows removed."""
    conn = _connect(read_only=False)
    try:
        cur = conn.execute("DELETE FROM leadership WHERE company = ?", (company,))
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def replace_company(company: str, leaders: Sequence[Mapping[str, Any]]) -> int:
    """Atomically refresh a company's leaders in ONE transaction, preserving
    history via a **soft delete**.

    Rather than hard-deleting, we flag the company's existing current rows as
    ``is_active = 0`` (historical) and INSERT the freshly extracted leaders as
    ``is_active = 1`` (current). This keeps an audit trail of who joined or left
    across re-scrapes. The scraper mines pages on many threads but writes once,
    here — doing the soft-delete + bulk INSERT in a single transaction on a single
    connection (instead of many small commits from worker threads) keeps the write
    atomic and avoids SQLite "database is locked" contention (.claudecode.md §3).
    Returns the number of new current rows inserted.
    """
    columns = ", ".join(_LEADER_COLUMNS)
    placeholders = ", ".join(f":{c}" for c in _LEADER_COLUMNS)
    insert_sql = f"INSERT INTO leadership ({columns}) VALUES ({placeholders})"
    rows = [{c: dict(leader).get(c) for c in _LEADER_COLUMNS} for leader in leaders]

    conn = _connect(read_only=False)
    try:
        conn.execute("BEGIN")
        conn.execute(
            "UPDATE leadership SET is_active = 0 WHERE company = ? AND is_active = 1",
            (company,),
        )
        if rows:
            conn.executemany(insert_sql, rows)  # is_active defaults to 1 (current)
        conn.commit()
        return len(rows)
    except sqlite3.Error:
        conn.rollback()
        raise
    finally:
        conn.close()


def insert_metrics(metrics: Mapping[str, Any]) -> None:
    """Record one collection run's performance/FinOps metrics (§3/§5).

    Accepts a plain mapping; unknown keys are ignored and missing keys become
    NULL, mirroring ``insert_leaders``. ``created_at`` defaults in the schema.
    """
    columns = ", ".join(_METRIC_COLUMNS)
    placeholders = ", ".join(f":{c}" for c in _METRIC_COLUMNS)
    sql = f"INSERT INTO system_metrics ({columns}) VALUES ({placeholders})"
    row = {c: dict(metrics).get(c) for c in _METRIC_COLUMNS}

    conn = _connect(read_only=False)
    try:
        conn.execute(sql, row)
        conn.commit()
    finally:
        conn.close()


def get_schema() -> str:
    """Return the pruned ``leadership`` table DDL for the LLM context.

    Only the user table is exposed — no FTS internals or sqlite_* system tables
    (.claudecode.md §5 schema pruning). The FTS5 query helper is documented
    inline so the agent knows how to do context search.
    """
    conn = _connect(read_only=True)
    try:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='leadership'"
        ).fetchone()
    finally:
        conn.close()

    ddl = row["sql"] if row else "(table not initialized)"
    return (
        f"{ddl}\n\n"
        "-- role_category is one of: 'C-Level', 'VP', 'Head'.\n"
        "-- HISTORY: rows are soft-deleted across re-scrapes. is_active = 1 is the\n"
        "--   current leadership; is_active = 0 is historical. ALWAYS filter\n"
        "--   WHERE is_active = 1 for present-day questions; only include inactive\n"
        "--   rows when the user explicitly asks about history / past leaders.\n"
        "-- For semantic/context search over bios, use the FTS5 table, e.g.:\n"
        "--   SELECT l.* FROM leadership l\n"
        "--   JOIN leadership_fts f ON f.rowid = l.id\n"
        "--   WHERE leadership_fts MATCH 'marketing growth' AND l.is_active = 1;"
    )


def execute_sql(query: str) -> dict[str, Any]:
    """Execute a read-only SELECT and return results, or an error message.

    Opens the DB in read-only mode (§4), so any write/DDL the LLM emits fails at
    the SQLite layer. On failure we return the error text rather than raising, so
    the agent can self-correct (the app wraps this in an <error> tag — §6).
    """
    conn = _connect(read_only=True)
    try:
        cur = conn.execute(query)
        columns = [d[0] for d in cur.description] if cur.description else []
        rows = [dict(r) for r in cur.fetchall()]
        return {"columns": columns, "rows": rows, "row_count": len(rows)}
    except sqlite3.Error as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
    print(f"Initialized database at {DB_PATH}")
    print(get_schema())
