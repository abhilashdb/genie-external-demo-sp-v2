"""LangGraph agent: supervisor (Claude via FM API) orchestrating a Genie tool.

Graph shape:

    START
      |
      v
    load_context  -- injects system prompt with user/dealership/summary
      |
      v
    maybe_summarize -- compresses messages into `summary` when thresholds hit
      |
      v
    supervisor (LLM with `ask_genie` tool)
      |
      +-- tool call?  --> genie_node --> supervisor
      |
      +-- no tool call --> END

State is checkpointed to Lakebase per `thread_id`. `genie_conv_id` is sticky
in state; the caller rotates it via the `/reset-genie` endpoint (Option D).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

from .. import flow_events, request_ctx
from ..config import settings
from .state import AgentState
from .tokens import count as count_tokens

log = logging.getLogger("genie_sp_demo_v2.agent")

# ----- Supervisor system prompt -----

_SUPERVISOR_SYSTEM = """You are a data analyst assistant for a car-dealership operator.
You have one or more Genie-space tools, each scoped to a distinct data domain.
Every space runs against Unity Catalog with row-level security (the SP identity
restricts results to the user's own dealership).

CONFIGURED GENIE SPACES — pick the right tool for each question:
{spaces_block}

ROUTING DISCIPLINE:

1. Match the user's question to ONE space by reading the scope descriptions
   above. Call only that space's tool (e.g. `ask_sales_genie` or
   `ask_crm_genie`). Never invent a tool name that isn't listed.

2. If the user's question **doesn't fit any configured space**, do NOT call
   any tool. Refuse with:
     "I'm not configured to answer questions on that topic. This assistant
      covers: <summarize each configured space in one clause>. You may need
      a different tool or data source for that question."

3. **Ambiguity between spaces** — if a question could match two spaces (e.g.
   the word "deals" could mean completed sales OR pipeline opportunities),
   do NOT guess. Ask the user to confirm, naming each option with its space.
   Example:
     "'Deals' could mean two things:
        a) **Completed sales** (Sales & Service space) — who bought what,
           when, for how much
        b) **Pipeline opportunities** (CRM space) — open deals in the funnel,
           expected close, forecasts
      Which would you like?"
   Forward only after the user picks.

4. Follow-ups ("sort by desc", "what about last quarter?") stay on the same
   space as the prior turn — do NOT switch spaces mid-thread without a clear
   signal from the user. Each space maintains its own conversation context.

5. Pass the user's question to the chosen tool **verbatim**. Allowed edits:
   fix typos, resolve pronouns, trim preamble. Do NOT invent metrics,
   groupings, columns, or time windows.

6. No space does root-cause / "why X is declining" analysis — Genie's agent
   RCA mode is not API-exposed. For "why" questions, offer factual data cuts
   the user can inspect themselves.

HOW TO USE `ask_genie`:
- **Pass the user's question verbatim**, or as close to it as possible. Do
  not invent specifics (columns, groupings, time windows, metrics) that the
  user did not mention. Your job is to route, not to redesign the query.
- Allowed changes: fix typos, resolve pronouns ("those" → the subject from
  the rolling summary), and trim obvious non-question preamble. That is all.
- Short follow-ups like "sort by revenue desc" or "break that down by region"
  should be forwarded as-is — Genie maintains its own conversation context.
- Genie answers **factual data questions only** — what, how many, who, when,
  where, ranked-by, grouped-by. It does NOT do root-cause analysis or
  "why X is declining" reasoning (that feature exists only in Genie's UI
  agent mode, which this app cannot call).

WHEN THE USER'S QUESTION IS TOO VAGUE OR IS AN RCA/"WHY" QUESTION
(e.g., "how can I improve my sales?", "why are my sales down?",
"tell me about my business"):
- Do NOT call `ask_genie` with a guessed interpretation. Do NOT pick metrics
  for them, and do NOT attempt root-cause analysis yourself.
- Reply to the user with 2–3 concrete **factual** data cuts they can pick
  from, and let them inspect the data. Example:
    "I can pull factual data but I can't do root-cause analysis here.
     A few angles that might help — which would you like to see?
     1. Top salespeople by revenue and deals closed, last 90 days
     2. Sales trend by month for the last 6 months
     3. Vehicles with declining sales vs prior period
    Reply with a number or describe what you want to see."
- Only call `ask_genie` after the user picks a specific direction.

FOR GREETINGS / META / CLARIFICATIONS:
- Answer directly without calling the tool.

AFTER A TOOL RESULT:
- Write a concise prose summary (2–4 sentences) of what the data shows.
- **Do NOT repeat the table, SQL, or raw tool output** — the UI renders those
  separately below your message.
- Surface the headline finding (top value, total, trend) and any caveat.
- Never invent data. If Genie returned 0 rows or an error, say so plainly.

Context for this conversation:
- User: {local_user} ({dealership})
- Rolling summary of earlier turns: {summary}
""".strip()


# ----- Nodes -----


async def load_context_node(state: AgentState) -> Dict[str, Any]:
    """Inject / refresh the supervisor system prompt with current context.

    We always keep exactly one SystemMessage at the head of `messages`, rebuilt
    each turn so the summary stays fresh. The `add_messages` reducer will
    replace a SystemMessage with the same id.
    """
    session_id = request_ctx.get_session_id()
    await flow_events.publish(
        session_id,
        step="agent_load_context",
        status="ok",
        title="Loaded thread context",
        detail=f"summary_len={len(state.get('summary', '') or '')} chars",
        payload={
            "has_summary": bool(state.get("summary")),
            "genie_conv_id": state.get("genie_conv_id"),
        },
    )

    spaces = settings.genie_spaces()
    if spaces:
        spaces_block = "\n\n".join(
            f"- **{meta['label']}** (tool: `ask_{key}_genie`)\n"
            f"  Scope: {meta['scope'] or '(no scope configured)'}"
            for key, meta in spaces.items()
        )
    else:
        spaces_block = "(no Genie spaces configured — refuse any data question)"

    system = SystemMessage(
        id="system-prompt",
        content=_SUPERVISOR_SYSTEM.format(
            local_user=state.get("local_user", "unknown"),
            dealership=state.get("dealership", "unknown"),
            summary=state.get("summary") or "(none yet — first few turns)",
            spaces_block=spaces_block,
        ),
    )
    # Clear last-turn scratch so a refusal / non-Genie reply doesn't carry the
    # previous turn's SQL + table through to the HTTP response.
    return {"messages": [system], "last_genie_results": []}


async def maybe_summarize_node(state: AgentState) -> Dict[str, Any]:
    """Compress old messages into `summary` when thresholds are exceeded."""
    session_id = request_ctx.get_session_id()
    messages: List[BaseMessage] = [
        m for m in state.get("messages", []) if not isinstance(m, SystemMessage)
    ]
    turns = len(messages)
    tokens = count_tokens(messages)

    await flow_events.publish(
        session_id,
        step="agent_memory",
        status="ok",
        title="Memory snapshot",
        detail=f"{turns} msgs, ~{tokens} tokens",
        payload={
            "turns": turns,
            "tokens": tokens,
            "turn_threshold": settings.summary_turn_threshold,
            "token_threshold": settings.summary_token_threshold,
            "has_summary": bool(state.get("summary")),
        },
    )

    if (
        turns < settings.summary_turn_threshold
        and tokens < settings.summary_token_threshold
    ):
        return {}

    keep = max(1, settings.summary_keep_last)
    head, tail = messages[:-keep], messages[-keep:]
    if not head:
        return {}

    old_summary = state.get("summary") or ""
    llm = _get_llm()
    prompt = [
        SystemMessage(
            content=(
                "You compress chat history into a dense bullet-form summary. "
                "Keep factual details that might matter for later turns "
                "(user goals, entities mentioned, prior findings). Drop "
                "pleasantries. Respond with the updated summary only."
            )
        ),
        HumanMessage(
            content=(
                f"Existing summary:\n{old_summary or '(empty)'}\n\n"
                f"New messages to fold in:\n"
                + "\n".join(_render_for_summary(m) for m in head)
            )
        ),
    ]
    await flow_events.publish(
        session_id,
        step="agent_summarize",
        status="pending",
        title="Compressing conversation",
        detail=f"folding {len(head)} msgs into summary (keeping last {keep})",
        payload={"folded": len(head), "kept": len(tail)},
    )
    resp = await llm.ainvoke(prompt)
    new_summary = (resp.content or "").strip() if hasattr(resp, "content") else str(resp)

    await flow_events.publish(
        session_id,
        step="agent_summarize",
        status="ok",
        title="Summary updated",
        detail=new_summary,
        payload={"summary_chars": len(new_summary)},
    )

    # Replace history: drop head, rebuild tail + placeholder system (load_context
    # node runs before next turn and will overwrite the system prompt).
    # To drop messages from state under `add_messages` reducer, we use RemoveMessage.
    from langchain_core.messages import RemoveMessage

    removals = [RemoveMessage(id=_msg_id(m)) for m in head if _msg_id(m)]
    return {"summary": new_summary, "messages": removals}


def _render_for_summary(m: BaseMessage) -> str:
    role = type(m).__name__.replace("Message", "").lower()
    content = m.content if isinstance(m.content, str) else json.dumps(m.content)[:400]
    return f"[{role}] {content}"


def _msg_id(m: BaseMessage) -> Optional[str]:
    mid = getattr(m, "id", None)
    return mid if isinstance(mid, str) else None


# ----- Genie tools (one per configured space) + node -----


@tool("ask_sales_genie")
def ask_sales_genie(question: str) -> str:
    """Query the **Sales & Service** Genie space for completed-sales and
    service-ticket questions.

    Use when the user asks about: completed vehicle sales (who bought what,
    when, for how much), sales totals by salesperson / model / dealership,
    service-ticket history, repair costs. NOT for pipeline, leads, forecasts.

    Pass the user's question verbatim. Do NOT invent columns, groupings, or
    time windows the user did not specify. Short follow-ups ("sort by revenue
    desc", "what about Q3?") are fine — Genie keeps its own context per space.
    """
    raise RuntimeError("routed through genie_node")


@tool("ask_crm_genie")
def ask_crm_genie(question: str) -> str:
    """Query the **CRM / Pipeline** Genie space for leads, open opportunities,
    forecasts, and sales-target questions.

    Use when the user asks about: open leads and where they are in the funnel,
    pipeline opportunities by stage / expected close, forecasts for upcoming
    months, sales targets vs achieved. NOT for completed sales history.

    Pass the user's question verbatim. Do NOT invent columns, groupings, or
    time windows the user did not specify.
    """
    raise RuntimeError("routed through genie_node")


# Map tool name → space key. Kept next to the tool defs so adding a space is
# a three-line change (add tool decorator, add entry here, add env vars).
_TOOL_TO_SPACE = {
    "ask_sales_genie": "sales",
    "ask_crm_genie": "crm",
}
_ALL_GENIE_TOOLS = [ask_sales_genie, ask_crm_genie]


def _bind_genie_tools() -> List[Any]:
    """Return only the Genie tools whose spaces are configured."""
    spaces = settings.genie_spaces()
    return [t for t in _ALL_GENIE_TOOLS if _TOOL_TO_SPACE.get(t.name) in spaces]


async def supervisor_node(state: AgentState) -> Dict[str, Any]:
    """LLM turn: decides which Genie tool (if any) to call, or reply directly."""
    session_id = request_ctx.get_session_id()
    llm = _get_llm().bind_tools(_bind_genie_tools())
    resp = await llm.ainvoke(state["messages"])

    tool_calls = getattr(resp, "tool_calls", None) or []
    await flow_events.publish(
        session_id,
        step="agent_supervisor",
        status="ok",
        title=(
            f"Supervisor → {len(tool_calls)} tool call(s)"
            if tool_calls
            else "Supervisor answered directly"
        ),
        detail=resp.content if isinstance(resp.content, str) else "",
        payload={
            "tool_calls": [
                {"name": tc.get("name"), "args": tc.get("args", {})}
                for tc in tool_calls
            ],
        },
    )
    return {"messages": [resp]}


async def genie_node(state: AgentState) -> Dict[str, Any]:
    """Execute ALL pending Genie tool calls against the appropriate spaces.

    The supervisor LLM can emit multiple tool calls in one turn (e.g. hitting
    both Sales and CRM). Each call routes to its space by tool name, reuses
    or creates that space's sticky conv id, and produces one ToolMessage
    (required so every tool_call_id has a matching response). Results are
    accumulated into `last_genie_results` for the HTTP layer and into the
    append-only `genie_results` for replay.
    """
    from ..genie_client import GenieClient, GenieError, extract_sql_and_text

    session_id = request_ctx.get_session_id()
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    if not tool_calls:
        return {}

    spaces = settings.genie_spaces()
    # Back-compat: fold legacy `genie_conv_id` into `genie_conv_ids["sales"]`.
    conv_ids = dict(state.get("genie_conv_ids") or {})
    if "sales" not in conv_ids and state.get("genie_conv_id"):
        conv_ids["sales"] = state["genie_conv_id"]

    tool_msgs: List[ToolMessage] = []
    fresh_results: List[Dict[str, Any]] = []

    for tc in tool_calls:
        tool_name = tc.get("name") or ""
        space_key = _TOOL_TO_SPACE.get(tool_name)
        question = (tc.get("args") or {}).get("question") or ""
        call_id = tc.get("id") or f"genie_call_{len(tool_msgs)}"

        if not space_key or space_key not in spaces:
            tool_msgs.append(
                ToolMessage(
                    content=f"Error: Genie space '{space_key}' is not configured.",
                    tool_call_id=call_id,
                    name=tool_name or "ask_genie",
                )
            )
            continue

        space = spaces[space_key]
        genie_conv_id = conv_ids.get(space_key)
        client = GenieClient(
            space_id=space["id"],
            sp_token=request_ctx.get_sp_token(),
            session_id=session_id,
        )
        try:
            if genie_conv_id:
                msg_id = await client.send_message(genie_conv_id, question)
                conv_id = genie_conv_id
            else:
                conv_id, msg_id = await client.start_conversation(question)

            message_json = await client.poll_message(conv_id, msg_id)
            sql, answer_text = extract_sql_and_text(message_json)
            rows: Optional[List[Dict[str, Any]]] = None
            columns: Optional[List[Dict[str, str]]] = None
            if sql:
                await client.publish_sql(sql)
                try:
                    qr = await client.get_query_result(conv_id, msg_id)
                    columns = qr.get("columns") or None
                    rows = qr.get("rows") or None
                except GenieError as e:
                    await flow_events.publish(
                        session_id,
                        step="error",
                        status="error",
                        title="Genie query-result fetch failed",
                        detail=str(e),
                        payload={"where": "genie_query_result", "space": space_key},
                    )

            tool_msgs.append(
                ToolMessage(
                    content=_format_tool_result(answer_text, sql, columns, rows),
                    tool_call_id=call_id,
                    name=tool_name,
                )
            )
            result = {
                "space": space_key,
                "space_label": space["label"],
                "conversation_id": conv_id,
                "message_id": msg_id,
                "tool_call_id": call_id,
                "answer_text": answer_text,
                "sql": sql,
                "rows": rows,
                "columns": columns,
            }
            conv_ids[space_key] = conv_id
            fresh_results.append(result)
        except GenieError as e:
            tool_msgs.append(
                ToolMessage(
                    content=f"Genie error: {e}",
                    tool_call_id=call_id,
                    name=tool_name,
                )
            )
            await flow_events.publish(
                session_id,
                step="error",
                status="error",
                title=f"Genie call failed ({space_key})",
                detail=str(e),
                payload={"where": "genie_node", "space": space_key},
            )
        finally:
            await client.close()

    return {
        "messages": tool_msgs,
        "genie_conv_ids": conv_ids,
        "genie_conv_id": conv_ids.get("sales"),  # legacy alias
        "last_genie_results": fresh_results,
        "genie_results": fresh_results,
    }


def _format_tool_result(
    answer_text: Optional[str],
    sql: Optional[str],
    columns: Optional[List[Dict[str, str]]],
    rows: Optional[List[Dict[str, Any]]],
) -> str:
    """Compact summary of the Genie result for the LLM to reason over.

    We intentionally do NOT include the SQL or the full tabular data here —
    the UI renders those separately from the structured response, and when
    they're present in the tool message the LLM tends to paste them into its
    reply verbatim. We hand the LLM just enough to describe the result.
    """
    parts: List[str] = []
    if answer_text:
        parts.append(answer_text.strip())
    if rows is None and not answer_text and not sql:
        return "(empty Genie response)"
    if rows is not None:
        if len(rows) == 0:
            parts.append("The query returned 0 rows.")
        else:
            col_names = [c.get("name", "") for c in (columns or [])]
            sample = rows[0] if rows else {}
            sample_pairs = ", ".join(
                f"{k}={sample[k]}" for k in col_names if k in sample
            )
            parts.append(
                f"The query returned {len(rows)} row(s) across columns: "
                f"{', '.join(col_names) or '(unknown)'}. "
                f"First-row sample: {sample_pairs or '(none)'}."
            )
    if sql:
        parts.append("(SQL is shown separately in the UI — do not restate it.)")
    return "\n\n".join(parts)


# ----- Routing -----


def route_after_supervisor(state: AgentState) -> str:
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    return "genie" if tool_calls else END


# ----- LLM factory (Claude via Databricks Foundation Model API) -----

_llm_singleton = None


def _get_llm():
    global _llm_singleton
    if _llm_singleton is not None:
        return _llm_singleton
    from databricks_langchain import ChatDatabricks

    _llm_singleton = ChatDatabricks(
        endpoint=settings.fm_endpoint_name,
        temperature=settings.fm_temperature,
        max_tokens=settings.fm_max_tokens,
    )
    return _llm_singleton


# ----- Compiled graph -----

_graph_singleton = None


def build_graph(checkpointer):
    g = StateGraph(AgentState)
    g.add_node("load_context", load_context_node)
    g.add_node("maybe_summarize", maybe_summarize_node)
    g.add_node("supervisor", supervisor_node)
    g.add_node("genie", genie_node)

    g.add_edge(START, "load_context")
    g.add_edge("load_context", "maybe_summarize")
    g.add_edge("maybe_summarize", "supervisor")
    g.add_conditional_edges("supervisor", route_after_supervisor, {"genie": "genie", END: END})
    g.add_edge("genie", "supervisor")

    return g.compile(checkpointer=checkpointer)


def get_graph():
    """Return the compiled graph, built once per process."""
    global _graph_singleton
    if _graph_singleton is None:
        from .. import lakebase

        _graph_singleton = build_graph(lakebase.get_checkpointer())
    return _graph_singleton


def reset_graph() -> None:
    """For tests / when the checkpointer is re-initialized."""
    global _graph_singleton
    _graph_singleton = None
