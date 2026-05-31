"""Shared pytest fixtures.

Living at the project root, this file also ensures the root is on ``sys.path`` so
``import src`` resolves under the src/ layout without an installed package.

Tests cover the deterministic layers only — schema/SQL, security guards, and the
pure text helpers. They never call a real LLM or hit the network: that keeps the
suite fast and reproducible, and the spec's "no mocked completions" rule is about
the running product, not its unit tests.
"""

from __future__ import annotations

import pytest

from src import database


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """An initialized, isolated SQLite database for one test.

    Points ``database.DB_PATH`` at a tmp file (``_connect`` reads the module
    global at call time, so this redirects every connection) and runs the schema.
    Yields the ``database`` module so tests can call its functions directly.
    """
    db_file = tmp_path / "test_company_data.db"
    monkeypatch.setattr(database, "DB_PATH", db_file)
    database.init_db()
    return database
