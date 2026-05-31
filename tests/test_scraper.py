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
