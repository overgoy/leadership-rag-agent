"""Tests for the SQLite layer: schema, indexes, FTS5, and read-only security."""

from __future__ import annotations

ALICE = {
    "company": "acme.com",
    "name": "Alice Stone",
    "role": "Chief Technology Officer",
    "role_category": "C-Level",
    "department": "Engineering",
    "location": "Berlin",
    "bio": "Leads platform engineering and growth infrastructure.",
    "linkedin_url": "https://linkedin.com/in/alice",
    "source_url": "https://acme.com/team",
}
BOB = {
    "company": "other.com",
    "name": "Bob Lin",
    "role": "VP Marketing",
    "role_category": "VP",
    "department": "Marketing",
    "source_url": "https://other.com/about",
}


def test_init_creates_schema_and_indexes(temp_db):
    res = temp_db.execute_sql(
        "SELECT name FROM sqlite_master WHERE type='index' "
        "AND name LIKE 'idx_leadership%'"
    )
    names = {r["name"] for r in res["rows"]}
    assert names == {
        "idx_leadership_role",
        "idx_leadership_department",
        "idx_leadership_category",
        "idx_leadership_company",
        "idx_leadership_active",
    }


def test_get_schema_is_pruned(temp_db):
    schema = temp_db.get_schema()
    assert "CREATE TABLE leadership" in schema
    # role_category guidance + FTS usage hint are included for the agent...
    assert "role_category" in schema
    assert "leadership_fts MATCH" in schema
    # ...but no system/internal tables leak into the LLM context (§5).
    assert "sqlite_master" not in schema


def test_insert_and_select(temp_db):
    assert temp_db.insert_leaders([ALICE, BOB]) == 2
    res = temp_db.execute_sql(
        "SELECT name, role_category FROM leadership ORDER BY name"
    )
    assert res["row_count"] == 2
    assert res["columns"] == ["name", "role_category"]
    assert res["rows"][0]["name"] == "Alice Stone"


def test_insert_ignores_unknown_and_fills_missing_with_null(temp_db):
    temp_db.insert_leaders([{"name": "Carol", "company": "acme.com", "junk": "x"}])
    res = temp_db.execute_sql(
        "SELECT name, role, bio FROM leadership WHERE name='Carol'"
    )
    row = res["rows"][0]
    assert row["name"] == "Carol"
    assert row["role"] is None and row["bio"] is None  # missing -> NULL, no crash


def test_clear_company_is_scoped(temp_db):
    temp_db.insert_leaders([ALICE, BOB])
    removed = temp_db.clear_company("acme.com")
    assert removed == 1
    res = temp_db.execute_sql("SELECT company FROM leadership")
    assert [r["company"] for r in res["rows"]] == ["other.com"]


def test_fts_match_finds_bio_keyword(temp_db):
    temp_db.insert_leaders([ALICE, BOB])
    res = temp_db.execute_sql(
        "SELECT l.name FROM leadership l "
        "JOIN leadership_fts f ON f.rowid = l.id "
        "WHERE leadership_fts MATCH 'growth'"
    )
    assert [r["name"] for r in res["rows"]] == ["Alice Stone"]


def test_fts_stays_in_sync_after_delete(temp_db):
    temp_db.insert_leaders([ALICE])
    temp_db.clear_company("acme.com")  # fires the AFTER DELETE trigger
    res = temp_db.execute_sql(
        "SELECT count(*) AS n FROM leadership_fts WHERE leadership_fts MATCH 'growth'"
    )
    assert res["rows"][0]["n"] == 0


def test_read_only_blocks_writes(temp_db):
    temp_db.insert_leaders([ALICE])
    res = temp_db.execute_sql("DELETE FROM leadership")
    assert "error" in res
    assert "readonly" in res["error"].lower()
    # data untouched
    count = temp_db.execute_sql("SELECT count(*) AS n FROM leadership")
    assert count["rows"][0]["n"] == 1


def test_bad_sql_returns_error_not_raise(temp_db):
    res = temp_db.execute_sql("SELECT * FROM no_such_table")
    assert "error" in res
    assert "no_such_table" in res["error"]
