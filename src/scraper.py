"""Web search (Tavily) and LLM structured extraction of company leadership.

This is the data-collection layer. It is independent of the UI and only talks to
the database through ``database.insert_leaders`` (see .claudecode.md §2
"Separation of Concerns").

Pipeline for a company URL:
  1. Tavily search for the company's leadership/team pages (with raw content).
  2. For each page, truncate raw text to 15k chars (§3) and ask a cheap "mini"
     model (§5) to extract leaders as structured JSON, validated by pydantic.
  3. Keep only C-level / VP / Head roles (§2 target scope); attach the page URL
     as provenance (§2); dedupe by name; write to SQLite.

Resilience (.claudecode.md §3): Optional fields prevent validation crashes,
raw text is truncated to 15k chars, and litellm uses ``num_retries=3``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from urllib.parse import urlparse

import litellm
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from tavily import TavilyClient

from src import database

load_dotenv()

# Structured logging instead of bare prints, so collection runs emit timestamped,
# level-tagged records (works cleanly under `make collect` and in containers).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

# §5 FinOps: cost-effective "mini" model by default; overridable via env.
EXTRACTION_MODEL = os.getenv("EXTRACTION_MODEL", "gpt-4o-mini")

# §3: cap raw web text to avoid token overflow.
MAX_TEXT_CHARS = 15_000

# §5: bound generation cost. Higher than the agent's limit because one page may
# describe a whole leadership team with bios.
EXTRACTION_MAX_TOKENS = 4_000

# How many search results to mine per company.
MAX_RESULTS = 6

# §3 Performance / scalability: mine pages concurrently. Extraction and
# verification are network-bound LLM calls, so worker threads overlap their
# latency without needing async. Bounded to avoid hammering API rate limits;
# overridable via env for large domains.
MAX_WORKERS = int(os.getenv("SCRAPER_MAX_WORKERS", "8"))

# Allowed leadership tiers (§2 target scope). Anything else is dropped.
ROLE_CATEGORIES = ("C-Level", "VP", "Head")

# Board / governance titles to drop even if the model mislabels them as C-Level
# (§2 says ignore board members). Matched case-insensitively against the role.
_BOARD_TITLE_KEYWORDS = (
    "chair",  # Chair, Vice Chair, Chairman, Chairwoman
    "board",  # Board Member, Board of Directors
    "trustee",
    "treasurer",
    "secretary",
)


def _is_board_role(role: Optional[str]) -> bool:
    """True if a title is a board/governance role rather than an executive one."""
    if not role:
        return False
    low = role.lower()
    return any(kw in low for kw in _BOARD_TITLE_KEYWORDS)


class Leader(BaseModel):
    """A single extracted leader. Only ``name`` is required; every other field
    is Optional so a partial profile never fails validation (§3)."""

    name: str
    role: Optional[str] = Field(
        None, description="Full title, e.g. 'Chief Technology Officer'"
    )
    role_category: Optional[str] = Field(
        None, description="One of: 'C-Level', 'VP', 'Head'"
    )
    department: Optional[str] = Field(
        None, description="e.g. 'Marketing', 'Engineering'"
    )
    location: Optional[str] = Field(None, description="Where the person is based")
    bio: Optional[str] = Field(None, description="Short free-text profile / background")
    linkedin_url: Optional[str] = None


class LeadershipExtraction(BaseModel):
    """Container for the model's structured output."""

    leaders: list[Leader] = Field(default_factory=list)


_EXTRACTION_SYSTEM = (
    "You extract company leadership from web page text into structured JSON.\n"
    "<target>\n"
    "Extract ONLY people employed at the target company given in <company>.\n"
    "A page may mention executives from OTHER organizations — customers,\n"
    "partners, investors, vendors, or companies featured in case studies and\n"
    "quotes. Exclude all of them; include a person only if the text clearly\n"
    "shows they work at the target company itself.\n"
    "HEURISTIC: case studies, customer stories, testimonials, and interviews\n"
    "often quote an executive describing how THEIR company uses the target\n"
    "company's product. That person is a CUSTOMER and works at a DIFFERENT\n"
    "company — never extract them. If you cannot tell which company a person\n"
    "works at, leave them out.\n"
    "</target>\n"
    "<scope>\n"
    "Include ONLY: C-level executives (CEO, CTO, CFO, CMO, COO, etc.),\n"
    "Vice Presidents (any VP/SVP/EVP), and Heads of departments.\n"
    "EXCLUDE: junior staff, individual contributors, advisors, investors, and\n"
    "all BOARD / governance roles — Chair, Vice Chair, Chairman, Board Member,\n"
    "Board of Directors, Trustee, Treasurer, and Secretary. These are NOT\n"
    "C-level executives even though they sound senior; omit them entirely.\n"
    "</scope>\n"
    "<classification>\n"
    "Set role_category to exactly one of 'C-Level', 'VP', or 'Head' based on the\n"
    "person's title. If a person does not fit one of these tiers, omit them.\n"
    "</classification>\n"
    "<rules>\n"
    "- Only use facts present in the provided text; never invent people or bios.\n"
    "- Leave any unknown field null rather than guessing.\n"
    "- Return an empty list if the text contains no qualifying leaders.\n"
    "</rules>"
)

# §5: bound the verification call; it only returns a small JSON verdict.
VERIFY_MAX_TOKENS = 200

_VERIFY_SYSTEM = (
    "You verify, using ONLY the provided text, whether a named person is an\n"
    "employee/executive of the TARGET company.\n"
    "Answer works_at_target=true only if the text clearly shows the person works\n"
    "at the target company. Answer false if the text indicates (or even merely\n"
    "suggests) they belong to a DIFFERENT organization — for example a customer,\n"
    "partner, investor, or vendor quoted in a case study or testimonial. When in\n"
    "doubt, answer false."
)


class _Verification(BaseModel):
    """Result of the employment-verification pass."""

    works_at_target: bool
    reason: Optional[str] = None


def _usage(resp) -> tuple[int, float]:
    """Extract (total_tokens, estimated_cost_usd) from a litellm response for the
    observability layer (§5 FinOps). Best-effort — never raises, since metrics
    must not be able to break a collection run."""
    try:
        tokens = int(resp.usage.total_tokens or 0)
    except Exception:  # noqa: BLE001
        tokens = 0
    try:
        cost = float(litellm.completion_cost(completion_response=resp) or 0.0)
    except Exception:  # noqa: BLE001 — unknown model / pricing → treat as 0
        cost = 0.0
    return tokens, cost


def verify_employment(
    name: str, role: Optional[str], text: str, company: str
) -> tuple[bool, int, float]:
    """Skeptical second pass: confirm a candidate actually works at the target
    company before storing them. Fails open (keeps the candidate) on LLM error so
    a transient failure never silently drops a real leader.

    Returns ``(works_at_target, tokens_used, cost_usd)`` so the caller can roll
    the verification spend into the run's metrics."""
    snippet = text[:MAX_TEXT_CHARS]
    user_msg = (
        f"<company>{company}</company>\n"
        f"<person><name>{name}</name><title>{role or 'unknown'}</title></person>\n"
        "<page_text>\n"
        f"{snippet}\n"
        "</page_text>\n"
        "Does this person work at the target company?"
    )
    try:
        resp = litellm.completion(
            model=EXTRACTION_MODEL,
            messages=[
                {"role": "system", "content": _VERIFY_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            response_format=_Verification,
            num_retries=3,  # §3: retry transient errors / HTTP 429 rate limits
            retry_strategy="exponential_backoff_retry",  # back off on 429s
            max_tokens=VERIFY_MAX_TOKENS,
            temperature=0,
        )
        tokens, cost = _usage(resp)
        verdict = _Verification.model_validate_json(
            resp.choices[0].message.content or "{}"
        )
        return verdict.works_at_target, tokens, cost
    except Exception as exc:  # noqa: BLE001 — keep on error (fail open)
        logger.warning("verification error for %s: %s", name, exc)
        return True, 0, 0.0


def _company_from_url(url: str) -> str:
    """Derive a stable company identifier from a URL/domain (e.g. 'robinhood.com')."""
    netloc = urlparse(url if "://" in url else f"https://{url}").netloc or url
    return netloc.lower().removeprefix("www.")


def search_company(company: str) -> list[dict]:
    """Search the web for the company's leadership pages, returning a list of
    ``{"url", "content"}`` dicts with raw page text.

    The search is anchored to the company's own domain via ``include_domains`` so
    we don't pick up unrelated companies that share a name (homonyms). We try the
    own-domain pass first and only fall back to an unrestricted search if the
    site yields nothing usable.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set (see .env.example).")

    client = TavilyClient(api_key=api_key)
    query = f"{company} leadership team executives CEO CTO CFO VP heads"

    def _run(include_domains: Optional[list[str]]) -> list[dict]:
        resp = client.search(
            query=query,
            max_results=MAX_RESULTS,
            include_raw_content=True,
            search_depth="advanced",
            include_domains=include_domains,
        )
        pages: list[dict] = []
        for result in resp.get("results", []):
            text = result.get("raw_content") or result.get("content") or ""
            if text.strip():
                pages.append({"url": result.get("url", ""), "content": text})
        return pages

    pages = _run(include_domains=[company])
    if not pages:
        logger.info("no pages on %s; falling back to open search", company)
        pages = _run(include_domains=None)
    return pages


def extract_leaders(
    text: str, source_url: str, company: str
) -> tuple[list[dict], dict]:
    """Extract qualifying leaders from one page's text.

    Returns ``(leaders, stats)``. ``leaders`` are DB-ready dicts; ``company`` and
    ``source_url`` are attached here (not taken from the model) so provenance is
    trustworthy, never hallucinated (§2). ``stats`` carries this page's metrics
    (candidates proposed/verified, tokens, cost) for the observability layer.

    On any model/validation error an empty result is returned so one bad page
    can't abort the whole collection (§3).
    """
    stats = {
        "candidates_extracted": 0,
        "candidates_verified": 0,
        "tokens": 0,
        "cost": 0.0,
    }
    snippet = text[:MAX_TEXT_CHARS]
    user_msg = (
        f"<company>{company}</company>\n"
        f"<source_url>{source_url}</source_url>\n"
        "<page_text>\n"
        f"{snippet}\n"
        "</page_text>\n"
        "Extract the qualifying leaders as JSON."
    )

    try:
        resp = litellm.completion(
            model=EXTRACTION_MODEL,
            messages=[
                {"role": "system", "content": _EXTRACTION_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            response_format=LeadershipExtraction,
            num_retries=3,  # §3: retry transient errors / HTTP 429 rate limits
            retry_strategy="exponential_backoff_retry",  # back off on 429s
            max_tokens=EXTRACTION_MAX_TOKENS,
            temperature=0,
        )
        tokens, cost = _usage(resp)
        stats["tokens"] += tokens
        stats["cost"] += cost
        content = resp.choices[0].message.content or "{}"
        extraction = LeadershipExtraction.model_validate(json.loads(content))
    except (ValidationError, json.JSONDecodeError, KeyError, IndexError) as exc:
        logger.warning("extraction failed for %s: %s", source_url, exc)
        return [], stats
    except Exception as exc:  # network/LLM errors — skip this page, keep going
        logger.warning("LLM error for %s: %s", source_url, exc)
        return [], stats

    stats["candidates_extracted"] = len(extraction.leaders)
    leaders: list[dict] = []
    for leader in extraction.leaders:
        if leader.role_category not in ROLE_CATEGORIES:
            continue  # enforce §2 scope even if the model over-includes
        if _is_board_role(leader.role):
            continue  # drop board/governance roles (§2)
        ok, tokens, cost = verify_employment(leader.name, leader.role, snippet, company)
        stats["tokens"] += tokens
        stats["cost"] += cost
        if not ok:
            logger.info("skipped (not %s): %s (%s)", company, leader.name, leader.role)
            continue  # skeptical second pass: not an employee of the target
        row = leader.model_dump()
        row["company"] = company
        row["source_url"] = source_url
        leaders.append(row)

    stats["candidates_verified"] = len(leaders)
    return leaders, stats


def collect(url: str) -> int:
    """End-to-end collection for a company URL. Initializes the DB, replaces any
    existing rows for this company in one atomic transaction, and records run
    metrics. Returns the number of leaders stored."""
    company = _company_from_url(url)
    logger.info("Collecting leadership for: %s", company)
    started = time.perf_counter()

    database.init_db()
    pages = search_company(company)
    logger.info("found %d pages to mine", len(pages))

    # Mine pages concurrently (§3 scalability). Each page's extraction +
    # employment verification is an independent, network-bound unit of work, so
    # threads overlap their LLM latency. Bounded by MAX_WORKERS to stay within API
    # rate limits; per-call litellm retries handle transient 429s with backoff.
    # Results are kept in page order (indexed slots) so dedupe stays deterministic
    # regardless of completion order.
    per_page: list[list[dict]] = [[] for _ in pages]
    page_stats: list[dict] = [{} for _ in pages]
    if pages:
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(pages))) as pool:
            futures = {
                pool.submit(extract_leaders, page["content"], page["url"], company): idx
                for idx, page in enumerate(pages)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    per_page[idx], page_stats[idx] = future.result()
                except Exception as exc:  # one bad page can't abort the run (§3)
                    logger.warning("page mining failed: %s", exc)

    # Dedupe by name in page order — first occurrence wins.
    seen: set[str] = set()
    collected: list[dict] = []
    for leaders in per_page:
        for leader in leaders:
            key = leader["name"].strip().lower()
            if key and key not in seen:
                seen.add(key)
                collected.append(leader)

    # One atomic write from the main thread (§3): DELETE + bulk INSERT in a single
    # transaction, instead of many small commits from worker threads.
    inserted = database.replace_company(company, collected)
    logger.info("stored %d leaders for %s", inserted, company)

    # Telemetry: roll up per-page stats and record one metrics row for this run.
    duration = time.perf_counter() - started
    metrics = {
        "company": company,
        "duration_seconds": round(duration, 3),
        "pages_mined": len(pages),
        "candidates_extracted": sum(
            s.get("candidates_extracted", 0) for s in page_stats
        ),
        "candidates_verified": sum(s.get("candidates_verified", 0) for s in page_stats),
        "tokens_used": sum(s.get("tokens", 0) for s in page_stats),
        "estimated_cost_usd": round(sum(s.get("cost", 0.0) for s in page_stats), 6),
    }
    database.insert_metrics(metrics)
    logger.info(
        "metrics: %d pages, %d candidates -> %d verified, %d tokens, $%.6f, %.3fs",
        metrics["pages_mined"],
        metrics["candidates_extracted"],
        metrics["candidates_verified"],
        metrics["tokens_used"],
        metrics["estimated_cost_usd"],
        metrics["duration_seconds"],
    )
    return inserted


def main(argv: Optional[list[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        logger.error("Usage: python -m src.scraper <company_url>")
        return 2
    collect(argv[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
