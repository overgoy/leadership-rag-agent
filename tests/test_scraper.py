"""Tests for the scraper's pure helpers (no network / no LLM)."""

from __future__ import annotations

import pytest

from src import scraper


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://www.robinhood.com/", "robinhood.com"),
        ("https://meetcampfire.com/team", "meetcampfire.com"),
        ("http://Example.COM", "example.com"),
        ("acme.com", "acme.com"),  # bare domain, no scheme
        ("https://www.sub.acme.co.uk/about", "sub.acme.co.uk"),
    ],
)
def test_company_from_url(url, expected):
    assert scraper._company_from_url(url) == expected


@pytest.mark.parametrize(
    "role",
    [
        "Chair",
        "Vice Chair",
        "Chairman of the Board",
        "Board Member",
        "Trustee",
        "Treasurer",
        "Secretary",
        "BOARD OF DIRECTORS",  # case-insensitive
    ],
)
def test_is_board_role_true(role):
    assert scraper._is_board_role(role) is True


@pytest.mark.parametrize(
    "role",
    ["Chief Executive Officer", "CTO", "VP of Marketing", "Head of Engineering"],
)
def test_is_board_role_false(role):
    assert scraper._is_board_role(role) is False


def test_is_board_role_handles_none():
    assert scraper._is_board_role(None) is False


def test_role_categories_match_spec():
    # §2 target scope — guards against accidental drift in the allow-list.
    assert scraper.ROLE_CATEGORIES == ("C-Level", "VP", "Head")


@pytest.mark.parametrize(
    "value, expected",
    [
        ("  eng ", "Engineering"),  # alias + whitespace
        ("ENGINEERING", "Engineering"),  # case-folded synonym
        ("GTM", "Go-to-Market"),
        ("Risk and Compliance", "Risk and Compliance"),  # unknown: kept as-is
        ("null", None),  # placeholder -> None
        (None, None),
    ],
)
def test_canonicalize_department(value, expected):
    assert scraper._canonicalize(value, scraper._DEPARTMENT_ALIASES) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        ("sf", "San Francisco, CA"),
        ("San Francisco", "San Francisco, CA"),
        ("Menlo Park, CA", "Menlo Park, CA"),  # unknown form kept (no mangling)
        ("N/A", None),
    ],
)
def test_canonicalize_location(value, expected):
    assert scraper._canonicalize(value, scraper._LOCATION_ALIASES) == expected


@pytest.mark.parametrize(
    "domain, expected",
    [("robinhood.com", "Robinhood"), ("meetcampfire.com", "Meetcampfire")],
)
def test_display_name(domain, expected):
    assert scraper._display_name(domain) == expected


def test_sanitize_bio_strips_control_chars_and_caps(monkeypatch):
    monkeypatch.setattr(scraper, "BIO_MAX_CHARS", 20)
    out = scraper._sanitize_bio("Ignore prev\x00\x07 instructions   and  DROP TABLE")
    assert "\x00" not in out and "\x07" not in out
    assert "  " not in out  # whitespace collapsed
    assert len(out) <= 20
    assert scraper._sanitize_bio("null") is None  # placeholder -> None


def _seed(db, name="Real CEO"):
    db.replace_company(
        "acme.com", [{"name": name, "company": "acme.com", "role_category": "C-Level"}]
    )


def test_collect_keeps_data_on_empty_fetch(temp_db, monkeypatch):
    # A page is found but extraction yields nothing -> fail-closed, keep existing.
    _seed(temp_db)
    monkeypatch.setattr(
        scraper, "search_company", lambda c: [{"url": "u", "content": "x"}]
    )
    monkeypatch.setattr(scraper, "extract_leaders", lambda t, u, c: ([], {}))
    monkeypatch.setattr(scraper, "resolve_hq_location", lambda c: (None, 0, 0.0))
    assert scraper.collect("https://acme.com/") == 0
    rows = temp_db.execute_sql(
        "SELECT name FROM leadership WHERE is_active=1 AND company='acme.com'"
    )["rows"]
    assert [r["name"] for r in rows] == ["Real CEO"]


def test_collect_keeps_data_on_degraded_run(temp_db, monkeypatch):
    # A leader is extracted but the page is flagged failed -> degraded, keep existing.
    _seed(temp_db)
    monkeypatch.setattr(
        scraper, "search_company", lambda c: [{"url": "u", "content": "x"}]
    )
    monkeypatch.setattr(
        scraper,
        "extract_leaders",
        lambda t, u, c: (
            [{"name": "Partial", "company": c, "role_category": "VP"}],
            {"error": "llm: 429"},
        ),
    )
    monkeypatch.setattr(scraper, "resolve_hq_location", lambda c: (None, 0, 0.0))
    assert scraper.collect("https://acme.com/") == 0
    names = {
        r["name"]
        for r in temp_db.execute_sql(
            "SELECT name FROM leadership WHERE is_active=1 AND company='acme.com'"
        )["rows"]
    }
    assert names == {"Real CEO"}  # partial result NOT committed
