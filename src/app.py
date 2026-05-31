"""Streamlit chat UI + ReAct Text-to-SQL agent over the leadership database.

This is the UI / agent layer. It never touches the scraper and reaches the data
only through ``database.execute_sql`` / ``database.get_schema`` (.claudecode.md §2
"Separation of Concerns").

Agent design (.claudecode.md §6):
- The model is given the pruned schema and a single ``execute_sql`` tool.
- It reasons inside <thinking>...</thinking> tags; that chain of thought is shown
  to the user in a collapsed ``st.expander`` rather than the main answer.
- ReAct guardrails: a hard cap on tool-calling iterations, a persona bounded to
  this dataset (off-topic questions are politely refused), and a SELECT-only
  guard on top of the read-only connection (§4 defense in depth).
- When a query fails, the SQLite error is fed back wrapped in an <error> tag so
  the model can self-correct.

Resilience / UX (.claudecode.md §3): the final answer is streamed token-by-token
via ``st.write_stream``; conversation memory lives in ``st.session_state``; the
DB schema read is cached with ``st.cache_data``; litellm uses ``num_retries=3``.

Security (.claudecode.md §4): all SQL runs through ``database.execute_sql`` which
opens the DB ``mode=ro``; any write/DDL the model emits fails at the SQLite layer.

FinOps (.claudecode.md §5): a cost-effective "mini" model by default, bounded
``max_tokens``, a pruned schema, and truncated tool results.
"""

from __future__ import annotations

import json
import logging
import os
import re

import litellm
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src import database

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

# §5 FinOps: default to a cost-effective model; override via env for harder runs.
AGENT_MODEL = os.getenv("AGENT_MODEL", "gpt-4o-mini")

# §6 ReAct guardrail: never loop forever on tool calls.
MAX_ITERATIONS = 5

# §5: bound the agent's generation cost.
AGENT_MAX_TOKENS = 1_000

# §5: cap rows handed back to the model so a broad query can't blow up context.
MAX_ROWS_TO_MODEL = 50

_THINKING_RE = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = "<thinking>"

# Single tool the agent may call: a read-only SELECT against the leadership DB.
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_sql",
            "description": (
                "Run ONE read-only SQLite SELECT against the leadership database "
                "and return the matching rows as JSON. The database is read-only; "
                "INSERT/UPDATE/DELETE/DDL will fail."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A single SQLite SELECT statement.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


@st.cache_data(show_spinner=False)
def _cached_schema() -> str:
    """Cache the pruned schema read so we don't re-query sqlite_master every turn
    (.claudecode.md §3). Cleared automatically when the app process restarts."""
    return database.get_schema()


def _build_system_prompt(schema: str) -> str:
    """Assemble the XML-tagged agent system prompt around the live schema (§6)."""
    return (
        "<role>\n"
        "You are a precise data analyst for a company-leadership database. You "
        "answer questions about company executives strictly from query results. "
        "Never invent people, titles, or facts.\n"
        "</role>\n"
        "<database_schema>\n"
        f"{schema}\n"
        "</database_schema>\n"
        "<tools>\n"
        "You have one tool, execute_sql(query). It runs a single read-only SELECT "
        "and returns rows as JSON.\n"
        "</tools>\n"
        "<guardrails>\n"
        "You ONLY help with questions about the company leadership in this "
        "database (who holds which role, counts by role/department, locations, "
        "bios, sources, etc.). If the user asks anything off-topic — general "
        "knowledge, coding help, opinions, or anything unrelated to these leaders "
        "— politely decline in one sentence, state what you can help with, and do "
        "NOT call any tool.\n"
        "</guardrails>\n"
        "<process>\n"
        "1. Reason step by step about the question and the SQL that answers it. "
        "Put this reasoning inside <thinking>...</thinking> tags.\n"
        "2. Call execute_sql with a SELECT. Filter by role_category "
        "('C-Level','VP','Head'), department, or company where relevant. For "
        "keyword/semantic search over bios, JOIN leadership_fts with MATCH.\n"
        "3. If a query returns an <error>, read it, fix the SQL, and retry.\n"
        "4. When you have enough data, write a concise plain-language answer and "
        "cite the source_url for each leader you mention so the user can verify.\n"
        "</process>\n"
        "<rules>\n"
        "- Only SELECT statements; never attempt to modify data.\n"
        "- If the data does not contain the answer, say so plainly — do not guess.\n"
        "- Ground every claim in query results.\n"
        "</rules>"
    )


def _is_select(query: str) -> bool:
    """Defense-in-depth (§4): allow only read statements before they reach SQLite."""
    head = query.lstrip().lstrip("(").lower()
    return head.startswith("select") or head.startswith("with")


def _run_tool(query: str) -> tuple[str, dict]:
    """Execute one agent SQL call. Returns (content_for_model, raw_result).

    On any failure the content is wrapped in an <error> tag so the model can
    self-correct (§6). Successful results are truncated to ``MAX_ROWS_TO_MODEL``.
    """
    if not _is_select(query):
        return (
            "<error>Only read-only SELECT statements are permitted.</error>",
            {"error": "non-select rejected"},
        )

    result = database.execute_sql(query)
    if "error" in result:
        return f"<error>{result['error']}</error>", result

    rows = result["rows"]
    truncated = rows[:MAX_ROWS_TO_MODEL]
    payload = {
        "columns": result["columns"],
        "rows": truncated,
        "row_count": result["row_count"],
    }
    if result["row_count"] > MAX_ROWS_TO_MODEL:
        payload["note"] = (
            f"showing first {MAX_ROWS_TO_MODEL} of {result['row_count']} rows"
        )
    return json.dumps(payload, default=str), result


def _visible_answer(raw: str) -> str:
    """Strip <thinking> from streamed content so only the answer reaches the UI.

    Removes complete <thinking>...</thinking> blocks and hides any trailing
    *unclosed* block — including a half-arrived opening tag like ``<thi`` — so no
    fragment of the reasoning ever flashes on screen. The result grows
    monotonically as more tokens arrive, which keeps it safe to diff for
    incremental ``st.write_stream`` output.
    """
    text = _THINKING_RE.sub("", raw)
    low = text.lower()

    # Hold back a fully-typed but unclosed <thinking>...
    i = low.rfind(_THINK_OPEN)
    if i != -1 and "</thinking>" not in low[i:]:
        return text[:i]

    # ...and hold back a trailing partial opening tag (e.g. "<", "<thi").
    for n in range(min(len(text), len(_THINK_OPEN) - 1), 0, -1):
        if low[-n:] == _THINK_OPEN[:n]:
            return text[:-n]
    return text


class _ThinkingFilter:
    """Streaming-safe <thinking> stripper: feed raw content deltas, get back only
    the newly-revealed answer text to emit."""

    def __init__(self) -> None:
        self._raw = ""
        self._emitted = 0

    def feed(self, delta: str) -> str:
        self._raw += delta
        visible = _visible_answer(self._raw)
        new = visible[self._emitted :]
        self._emitted = len(visible)
        return new


def run_agent(history: list[dict], steps: list[dict]):
    """Drive the ReAct loop, yielding the final answer's tokens for streaming.

    ``history`` is the running list of ``{"role", "content"}`` chat messages.
    Tool calls and the model's <thinking> are appended to ``steps`` (rendered in
    the reasoning expander); only the visible final answer is yielded, so the
    caller can pass this generator straight to ``st.write_stream`` (§3).
    """
    messages: list = [
        {"role": "system", "content": _build_system_prompt(_cached_schema())}
    ]
    messages += [{"role": m["role"], "content": m["content"]} for m in history]

    for _ in range(MAX_ITERATIONS):
        stream = litellm.completion(
            model=AGENT_MODEL,
            messages=messages,
            tools=_TOOLS,
            tool_choice="auto",
            stream=True,
            num_retries=3,  # §3 stability
            max_tokens=AGENT_MAX_TOKENS,
            temperature=0,
        )

        chunks = []
        flt = _ThinkingFilter()
        for chunk in stream:
            chunks.append(chunk)
            delta = chunk.choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:
                visible = flt.feed(piece)
                if visible:
                    yield visible  # only post-<thinking> answer text streams out

        msg = litellm.stream_chunk_builder(chunks, messages=messages).choices[0].message
        messages.append(msg)

        thinking = "\n".join(t.strip() for t in _THINKING_RE.findall(msg.content or ""))
        if thinking:
            steps.append({"type": "thinking", "text": thinking})

        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            return  # final answer already streamed above

        for call in tool_calls:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            query = (args.get("query") or "").strip()

            content, raw = _run_tool(query)
            steps.append({"type": "sql", "query": query, "result": raw})
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": content}
            )

    yield "\n\n_(Stopped after reaching the step limit — try narrowing the question.)_"


def _render_steps(steps: list[dict]) -> None:
    """Show the agent's chain of thought and SQL inside a collapsed expander (§6)."""
    if not steps:
        return
    with st.expander("🧠 Reasoning & SQL"):
        for step in steps:
            if step["type"] == "thinking":
                st.markdown(step["text"])
            else:
                st.code(step["query"], language="sql")
                result = step["result"]
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.caption(f"{result['row_count']} row(s)")
                    if result["rows"]:
                        st.dataframe(result["rows"], use_container_width=True)


def render_chat() -> None:
    """The 'Chat Assistant' view: the ReAct Text-to-SQL agent."""
    st.title("🧭 Company Leadership Agent")
    st.caption(
        "Ask about company executives. Answers are grounded in a local SQLite "
        "database and cite their sources."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []  # §3 conversation memory

    # Replay prior turns (with their reasoning expanders).
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                _render_steps(msg.get("steps", []))
            st.markdown(msg["content"])

    prompt = st.chat_input("e.g. Who are the C-level executives at meetcampfire.com?")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        reasoning_slot = st.container()  # reserve position ABOVE the answer
        steps: list[dict] = []
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]
        try:
            answer = st.write_stream(run_agent(history, steps))  # §3 token streaming
        except Exception as exc:  # noqa: BLE001 — surface, never crash the UI (§3)
            logger.exception("agent run failed")
            answer = f"Sorry, something went wrong: {exc}"
            st.markdown(answer)

        if isinstance(answer, list):  # st.write_stream returns a list if mixed types
            answer = "".join(str(a) for a in answer)
        if not answer.strip():
            answer = "(no answer)"
            st.markdown(answer)

        with reasoning_slot:
            _render_steps(steps)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "steps": steps}
    )


def _query(sql: str) -> list[dict]:
    """Run a read-only query for the dashboard, returning rows (or [] on error).

    Goes through the same read-only ``execute_sql`` path as the agent (§4); the
    dashboard's queries are static (no user input), so there's no injection risk.
    """
    result = database.execute_sql(sql)
    return [] if "error" in result else result["rows"]


def render_dashboard() -> None:
    """The 'System Insights & Dashboard' view: pipeline + FinOps telemetry."""
    st.title("📊 System Insights & Dashboard")
    st.caption("Pipeline performance and FinOps telemetry across all collection runs.")

    agg = _query(
        "SELECT COALESCE(SUM(estimated_cost_usd), 0) AS cost, "
        "COALESCE(SUM(tokens_used), 0) AS tokens, "
        "COALESCE(SUM(candidates_verified), 0) AS verified, "
        "COALESCE(SUM(candidates_extracted), 0) AS extracted "
        "FROM system_metrics"
    )[0]
    records = _query("SELECT COUNT(*) AS n FROM leadership WHERE is_active = 1")[0]["n"]
    extracted = agg["extracted"] or 0
    success_rate = (agg["verified"] / extracted * 100) if extracted else 0.0

    # KPI cards (§ observability).
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Infra Cost", f"${agg['cost']:,.4f}", help="Aggregated LLM spend")
    c2.metric(
        "Scraper Success Rate",
        f"{success_rate:.0f}%",
        help="Verified ÷ extracted candidates across all runs",
    )
    c3.metric("Total Records", f"{records:,}")
    c4.metric("LLM Tokens", f"{agg['tokens']:,}")

    runs = _query(
        "SELECT company, created_at, duration_seconds, pages_mined, "
        "candidates_extracted, candidates_verified, tokens_used, estimated_cost_usd "
        "FROM system_metrics ORDER BY created_at"
    )
    if not runs:
        st.info(
            "No collection runs recorded yet. Run `make collect URL=...` to populate "
            "telemetry."
        )
        return

    # Diagnostic charts.
    left, right = st.columns(2)
    with left:
        st.subheader("Pipeline runtime by company")
        runtimes = _query(
            "SELECT company, ROUND(AVG(duration_seconds), 2) AS avg_seconds "
            "FROM system_metrics GROUP BY company"
        )
        st.bar_chart(pd.DataFrame(runtimes).set_index("company"))
    with right:
        st.subheader("Leadership by role category")
        dist = _query(
            "SELECT role_category, COUNT(*) AS count FROM leadership "
            "WHERE is_active = 1 GROUP BY role_category"
        )
        if dist:
            st.bar_chart(pd.DataFrame(dist).set_index("role_category"))
        else:
            st.caption("No leadership records yet.")

    with st.expander("Recent collection runs"):
        st.dataframe(pd.DataFrame(runs), use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Company Leadership Agent", page_icon="🧭")
    view = st.sidebar.radio(
        "View",
        ("💬 Chat Assistant", "📊 System Insights & Dashboard"),
    )
    if view.startswith("📊"):
        render_dashboard()
    else:
        render_chat()


if __name__ == "__main__":
    main()
