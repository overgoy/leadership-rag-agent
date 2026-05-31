"""Lightweight agent eval: run the task's example questions through the REAL
ReAct agent and check each answer against ground truth read from the database.

This is NOT a pytest test — it makes real LLM calls (per the task rule "Real LLMs
only — no mocked completions"), so it lives outside tests/ to keep `make test`
offline and deterministic. Ground truth is derived from SQL at run time, so the
checks track whatever data is currently collected rather than hard-coded names.

Run:  python -m eval_agent          (needs OPENAI_API_KEY, populated data/ DB)
Exit code is non-zero if any case fails, so it can gate CI later.
"""

from __future__ import annotations

import src.app as app
from src import database

# Headless bypass: run_agent reads the schema via an @st.cache_data helper, which
# warns outside a Streamlit runtime. Point it at the plain DB call instead.
app._cached_schema = database.get_schema  # type: ignore[assignment]

ROBINHOOD = "robinhood.com"
CAMPFIRE = "meetcampfire.com"

_CEO = "(lower(role) LIKE '%ceo%' OR lower(role) LIKE '%chief executive%')"
_NEGATIVE = (
    "no ",
    "not ",
    "n't",
    "none",
    "no one",
    "unavailable",
    "not available",
    "not listed",
    "not specified",
    "isn't",
    "couldn't",
    "could not",
    "do not have",
    "does not",
    "unknown",
)


def ask(question: str) -> str:
    """Run one question through the real agent and return the final answer text."""
    steps: list[dict] = []
    history = [{"role": "user", "content": question}]
    return "".join(app.run_agent(history, steps)).strip()


def _one(sql: str):
    """First cell of the first row, or None."""
    res = database.execute_sql(sql)
    if "error" in res or not res["rows"]:
        return None
    return next(iter(res["rows"][0].values()))


def _says_no(answer: str) -> bool:
    low = answer.lower()
    return any(marker in low for marker in _NEGATIVE)


def check_ceo_name(answer: str) -> tuple[bool, str]:
    name = _one(
        f"SELECT name FROM leadership WHERE is_active=1 AND company='{ROBINHOOD}' "
        f"AND {_CEO} LIMIT 1"
    )
    if not name:
        return False, "no CEO in data to check against"
    surname = name.split()[-1]
    ok = surname.lower() in answer.lower()
    return ok, f"expected CEO surname '{surname}' (from '{name}')"


def check_vp_count(answer: str) -> tuple[bool, str]:
    count = _one(
        f"SELECT COUNT(*) FROM leadership WHERE is_active=1 AND company='{ROBINHOOD}' "
        "AND role_category='VP'"
    )
    ok = str(count) in answer
    return ok, f"expected VP count '{count}' to appear in the answer"


def check_marketing_head(answer: str) -> tuple[bool, str]:
    name = _one(
        f"SELECT name FROM leadership WHERE is_active=1 AND company='{ROBINHOOD}' "
        "AND lower(department) LIKE '%marketing%' LIMIT 1"
    )
    if name:
        surname = name.split()[-1]
        return surname.lower() in answer.lower(), f"expected marketing head '{name}'"
    # No marketing head in data: the agent must say so, not invent one.
    return _says_no(answer), "no marketing head in data — agent must say so, not guess"


def check_ceo_location(answer: str) -> tuple[bool, str]:
    loc = _one(
        f"SELECT location FROM leadership WHERE is_active=1 AND company='{CAMPFIRE}' "
        f"AND {_CEO} LIMIT 1"
    )
    if loc:
        # Match on the first token of the location (e.g. city) to stay lenient.
        token = loc.replace(",", " ").split()[0]
        return token.lower() in answer.lower(), f"expected CEO location '{loc}'"
    # Location unknown: the agent must admit it rather than fabricate a city.
    return _says_no(answer), "CEO location unknown — agent must admit it, not invent"


CASES = [
    (f"Who is the CEO of {ROBINHOOD}?", check_ceo_name),
    (f"How many VPs does {ROBINHOOD} have?", check_vp_count),
    (f"Who heads marketing at {ROBINHOOD}?", check_marketing_head),
    (f"Where is the CEO of {CAMPFIRE} based?", check_ceo_location),
]


def main() -> int:
    passed = 0
    print("Agent eval — 4 task questions against live data + real LLM\n")
    for question, check in CASES:
        answer = ask(question)
        ok, note = check(answer)
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {question}")
        print(f"       check : {note}")
        print(f"       answer: {answer[:200]}\n")
    total = len(CASES)
    print(f"Result: {passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
