"""Tests for the agent's pure helpers: the SQL guard, <thinking> stripping, and
the read-only tool wrapper. No real LLM calls."""

from __future__ import annotations

import json

import pytest

from src import app


@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM leadership",
        "  select name from leadership",
        "WITH x AS (SELECT 1) SELECT * FROM x",
        "(SELECT 1)",
    ],
)
def test_is_select_allows_reads(query):
    assert app._is_select(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "DELETE FROM leadership",
        "DROP TABLE leadership",
        "INSERT INTO leadership(name) VALUES ('x')",
        "UPDATE leadership SET name='x'",
        "PRAGMA writable_schema=1",
    ],
)
def test_is_select_rejects_writes(query):
    assert app._is_select(query) is False


def test_visible_answer_strips_complete_block():
    assert (
        app._visible_answer("<thinking>plan</thinking>Final answer") == "Final answer"
    )


def test_visible_answer_hides_unclosed_block():
    assert app._visible_answer("Lead-in <thinking>still reasoning") == "Lead-in "


def test_visible_answer_hides_partial_open_tag():
    # a half-arrived "<thi" must not flash on screen
    assert app._visible_answer("Answer <thi") == "Answer "


def test_visible_answer_keeps_literal_less_than():
    assert app._visible_answer("5 < 10 is true") == "5 < 10 is true"


@pytest.mark.parametrize(
    "deltas, expected",
    [
        (["<thi", "nking>r</thi", "nking>Hello", " world"], "Hello world"),
        (["<thinking>r</thinking>", "Answer ", "here"], "Answer here"),
        (["The value 5 ", "< 10 is true"], "The value 5 < 10 is true"),
        (["Just ", "plain ", "text"], "Just plain text"),
    ],
)
def test_thinking_filter_streams_only_answer(deltas, expected):
    flt = app._ThinkingFilter()
    assert "".join(flt.feed(d) for d in deltas) == expected


def test_run_tool_rejects_non_select():
    content, raw = app._run_tool("DROP TABLE leadership")
    assert content.startswith("<error>")
    assert "error" in raw


def test_run_tool_executes_select(temp_db):
    temp_db.insert_leaders([{"name": "Dana", "company": "acme.com"}])
    content, raw = app._run_tool("SELECT name FROM leadership")
    assert raw["row_count"] == 1
    payload = json.loads(content)
    assert payload["rows"][0]["name"] == "Dana"


def test_run_tool_wraps_sql_error(temp_db):
    content, raw = app._run_tool("SELECT * FROM nope")
    assert content.startswith("<error>") and content.endswith("</error>")
    assert "error" in raw


def test_run_tool_truncates_large_result(temp_db, monkeypatch):
    monkeypatch.setattr(app, "MAX_ROWS_TO_MODEL", 3)
    temp_db.insert_leaders(
        [{"name": f"P{i}", "company": "acme.com"} for i in range(10)]
    )
    content, raw = app._run_tool("SELECT name FROM leadership")
    payload = json.loads(content)
    assert raw["row_count"] == 10  # full count reported
    assert len(payload["rows"]) == 3  # but rows handed to the model are capped
    assert "note" in payload
