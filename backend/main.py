"""FastAPI app for Genie SP Demo v2.

v2 changes vs v1:
  - `/api/chat` runs through a LangGraph supervisor agent (Claude via FM API)
    that decides when to call Genie. State is checkpointed to Lakebase per
    `thread_id = f"{local_user}:{uuid}"`.
  - Rolling summary compresses old turns when thresholds are hit.
  - `genie_conv_id` is sticky on the thread (Option D); rotate via
    `POST /api/threads/{thread_id}/reset-genie`.
  - `app_conversations` lives in Lakebase (replaces v1 sqlite db.py).

Endpoints:
  POST /api/login
  POST /api/logout
  GET  /api/me
  POST /api/chat                             — run one agent turn
  POST /api/threads/{thread_id}/reset-genie  — Option D: rotate Genie conv
  GET  /api/conversations                    — list this user's threads
  GET  /api/conversations/{thread_id}        — fetch messages + summary for a thread
  GET  /api/events/stream                    — SSE transparency (unchanged)
  POST /api/dev/simulate-rate-limit          — unchanged dev toggle
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from pydantic import BaseModel, Field

from . import (
    databricks_auth,
    dev_flags,
    flow_events,
    lakebase,
    request_ctx,
    sql_client,
    tracing,
)
from .agent.graph import get_graph
from .config import settings
from .users import User, authenticate, get_user

log = logging.getLogger("genie_sp_demo_v2")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# --------------------------------------------------------------------
# App + static mount
# --------------------------------------------------------------------

app = FastAPI(title="Genie SP Demo v2", version="0.2.0")

_STATIC_DIR = Path(__file__).resolve().parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# --------------------------------------------------------------------
# Session cookie (unchanged from v1)
# --------------------------------------------------------------------

_SESSION_COOKIE = "session"
_SESSION_MAX_AGE = 60 * 60
_signer = URLSafeTimedSerializer(settings.app_session_secret, salt="genie-sp-demo-v2-session")


def _issue_session_cookie(response: Response, username: str, session_id: str) -> None:
    token = _signer.dumps({"username": username, "session_id": session_id})
    response.set_cookie(
        key=_SESSION_COOKIE,
        value=token,
        max_age=_SESSION_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=False,
        path="/",
    )


def _clear_session_cookie(response: Response) -> None:
    response.delete_cookie(_SESSION_COOKIE, path="/")


def _read_session(request: Request) -> Optional[Dict[str, str]]:
    raw = request.cookies.get(_SESSION_COOKIE)
    if not raw:
        return None
    try:
        data = _signer.loads(raw, max_age=_SESSION_MAX_AGE)
    except (BadSignature, SignatureExpired):
        return None
    if not isinstance(data, dict) or "username" not in data or "session_id" not in data:
        return None
    return data


class _AuthCtx:
    def __init__(self, user: User, session_id: str):
        self.user = user
        self.session_id = session_id


def get_current_ctx(request: Request) -> _AuthCtx:
    sess = _read_session(request)
    if sess is None:
        raise HTTPException(status_code=401, detail="not authenticated")
    user = get_user(sess["username"])
    if user is None:
        raise HTTPException(status_code=401, detail="unknown user")
    return _AuthCtx(user=user, session_id=sess["session_id"])


def _user_public(user: User) -> Dict[str, str]:
    return {
        "username": user.username,
        "dealership": user.dealership,
        "role": user.role,
        "sp_label": user.sp_label,
    }


# --------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    thread_id: Optional[str] = None


class ResetGenieRequest(BaseModel):
    # Which Genie space to reset. None = reset all configured spaces.
    space: Optional[str] = None


class SimulateRateLimitRequest(BaseModel):
    count: int = Field(..., ge=0, le=10)
    status: int = Field(default=429)


# --------------------------------------------------------------------
# Root + health
# --------------------------------------------------------------------


@app.get("/")
async def root() -> FileResponse:
    index = _STATIC_DIR / "index.html"
    if not index.exists():
        return JSONResponse(
            {"ok": True, "detail": "backend running; frontend not yet built"},
            status_code=200,
        )
    return FileResponse(str(index))


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "genie_space_configured": bool(settings.genie_space_id),
        "lakebase_configured": bool(settings.lakebase_pg_uri),
        "fm_endpoint": settings.fm_endpoint_name,
        "dbx_host": settings.dbx_host,
    }


@app.get("/api/tracing/experiment")
async def tracing_experiment(
    ctx: _AuthCtx = Depends(get_current_ctx),
) -> Dict[str, Any]:
    """Return the MLflow experiment URL on the Databricks workspace so the UI
    can open it in a new tab. Best-effort — returns `url: None` if the
    experiment isn't set up yet.
    """
    if not settings.mlflow_enabled:
        return {"enabled": False, "url": None}
    try:
        import mlflow
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        client = mlflow.MlflowClient()
        exp = client.get_experiment_by_name(settings.mlflow_experiment_name)
        if not exp:
            return {"enabled": True, "url": None, "detail": "experiment not found"}
        host = settings.dbx_host.rstrip("/")
        return {
            "enabled": True,
            "experiment_id": exp.experiment_id,
            "experiment_name": exp.name,
            "url": f"{host}/ml/experiments/{exp.experiment_id}",
            "uc_schema": (
                f"{settings.mlflow_trace_uc_catalog}."
                f"{settings.mlflow_trace_uc_schema}"
                if settings.mlflow_trace_uc_catalog and settings.mlflow_trace_uc_schema
                else None
            ),
            "uc_table_prefix": settings.mlflow_trace_uc_prefix,
        }
    except Exception as e:
        return {"enabled": True, "url": None, "detail": str(e)}


# --------------------------------------------------------------------
# Auth (unchanged)
# --------------------------------------------------------------------


@app.post("/api/login")
async def login(body: LoginRequest) -> Response:
    user = authenticate(body.username, body.password)
    if user is None:
        raise HTTPException(status_code=401, detail="invalid credentials")

    session_id = uuid.uuid4().hex
    response = JSONResponse(_user_public(user))
    _issue_session_cookie(response, username=user.username, session_id=session_id)

    await flow_events.publish(
        session_id,
        step="login",
        status="ok",
        title=f"User {user.username} logged in",
        detail=f"Role: {user.role} @ {user.dealership}",
        payload={"username": user.username, "dealership": user.dealership, "role": user.role},
    )
    redacted_id = (
        (
            settings.sp_northstar_client_id
            if user.sp_label == "northstar"
            else settings.sp_sunrise_client_id
        )[:8]
        + "…"
    )
    await flow_events.publish(
        session_id,
        step="sp_resolve",
        status="ok",
        title=f"Resolved service principal for {user.dealership}",
        detail=f"sp_label={user.sp_label}",
        payload={
            "sp_label": user.sp_label,
            "sp_client_id": redacted_id,
            "sp_display_name": user.dealership,
        },
    )
    return response


@app.post("/api/logout")
async def logout(request: Request) -> Response:
    sess = _read_session(request)
    if sess and sess.get("session_id"):
        dev_flags.clear(sess["session_id"])
    response = JSONResponse({"ok": True})
    _clear_session_cookie(response)
    return response


@app.get("/api/me")
async def me(ctx: _AuthCtx = Depends(get_current_ctx)) -> Dict[str, str]:
    return _user_public(ctx.user)


# --------------------------------------------------------------------
# Dev: rate-limit sim (kept for parity with v1)
# --------------------------------------------------------------------


@app.post("/api/dev/simulate-rate-limit")
async def simulate_rate_limit(
    body: SimulateRateLimitRequest,
    ctx: _AuthCtx = Depends(get_current_ctx),
) -> Dict[str, Any]:
    new_count = dev_flags.arm_rate_limit(ctx.session_id, body.count, body.status)
    await flow_events.publish(
        ctx.session_id,
        step="genie_rate_limit",
        status="pending",
        title=f"Armed: next {new_count} Genie attempts will return {body.status}",
        detail="Ask a question to trigger the retry flow.",
        payload={"armed": new_count, "status": body.status, "simulated_setup": True},
    )
    return {"armed": new_count, "status": body.status}


# --------------------------------------------------------------------
# SSE
# --------------------------------------------------------------------


@app.get("/api/events/stream")
async def events_stream(ctx: _AuthCtx = Depends(get_current_ctx)) -> StreamingResponse:
    generator = flow_events.subscribe(ctx.session_id)
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# --------------------------------------------------------------------
# Chat (agent-driven)
# --------------------------------------------------------------------


def _new_thread_id(local_user: str) -> str:
    return f"{local_user}:{uuid.uuid4().hex}"


@app.post("/api/chat")
async def chat(body: ChatRequest, ctx: _AuthCtx = Depends(get_current_ctx)) -> Dict[str, Any]:
    user = ctx.user
    session_id = ctx.session_id

    if not settings.genie_space_id:
        raise HTTPException(status_code=503, detail="GENIE_SPACE_ID not configured")
    if not settings.lakebase_pg_uri:
        raise HTTPException(status_code=503, detail="LAKEBASE_PG_URI not configured")

    thread_id = body.thread_id or _new_thread_id(user.username)
    is_new_thread = not body.thread_id

    # Exchange SP credentials for a fresh bearer (cached by sp_label).
    try:
        sp_token = await databricks_auth.get_sp_token(user.sp_label, session_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"token_exchange failed: {e}")

    # Record / bump the thread pointer row.
    await lakebase.upsert_conversation(
        thread_id=thread_id,
        local_user=user.username,
        sp_label=user.sp_label,
        first_message=body.message,
    )

    graph = get_graph()
    config = {
        "configurable": {"thread_id": thread_id},
        "run_name": f"chat/{user.username}/{thread_id.split(':')[-1][:8]}",
        "tags": {
            "thread_id": thread_id,
            "local_user": user.username,
            "sp_label": user.sp_label,
            "dealership": user.dealership,
        },
    }

    from langchain_core.messages import HumanMessage

    # Keep secrets OUT of graph state so they don't land in MLflow traces
    # or the Lakebase checkpoint. `request_ctx` is a ContextVar that scopes
    # to this asyncio task and is read from inside the nodes.
    request_ctx.set_sp_token(sp_token)
    request_ctx.set_session_id(session_id)

    inp: Dict[str, Any] = {
        "messages": [HumanMessage(content=body.message)],
        "local_user": user.username,
        "sp_label": user.sp_label,
        "dealership": user.dealership,
    }
    if is_new_thread:
        # Seed state with empty summary / no Genie conv on first turn.
        inp["summary"] = ""
        inp["genie_conv_id"] = None

    try:
        with tracing.chat_turn_span(
            thread_id=thread_id,
            username=user.username,
            dealership=user.dealership,
            sp_label=user.sp_label,
            user_message=body.message,
        ) as span_ctx:
            final_state = await graph.ainvoke(inp, config=config)
            span_ctx.set_outputs(
                answer_text=_last_ai_text(final_state.get("messages") or []),
                spaces_used=[
                    r.get("space") for r in (final_state.get("last_genie_results") or [])
                ],
            )
    except Exception as e:
        await flow_events.publish(
            session_id,
            step="error",
            status="error",
            title="Agent invocation failed",
            detail=str(e),
            payload={"where": "graph.ainvoke", "message": str(e)},
        )
        raise HTTPException(status_code=500, detail=str(e))

    # Persist the sales Genie conv id on the sidebar pointer row for now
    # (the row has a single `genie_conv_id` column — CRM conv ids live only
    # in LangGraph state). A future extension could split the column per space.
    conv_ids = final_state.get("genie_conv_ids") or {}
    sales_conv = conv_ids.get("sales") or final_state.get("genie_conv_id")
    if sales_conv:
        await lakebase.set_genie_conv_id(thread_id, sales_conv)

    # Extract final assistant message text + any Genie results from this turn.
    last = _last_ai_text(final_state.get("messages") or [])
    last_results = final_state.get("last_genie_results") or []

    await flow_events.publish(
        session_id,
        step="response",
        status="ok",
        title="Response ready",
        detail=(last or ""),
        payload={
            "thread_id": thread_id,
            "spaces_used": [r.get("space") for r in last_results],
        },
    )

    return {
        "thread_id": thread_id,
        "genie_conv_id": sales_conv,       # legacy alias; kept for UI
        "genie_conv_ids": conv_ids,         # per-space map
        "answer_text": last,
        "results": [
            {
                "space": r.get("space"),
                "space_label": r.get("space_label"),
                "sql": r.get("sql"),
                "rows": r.get("rows"),
                "columns": r.get("columns"),
            }
            for r in last_results
        ],
        "summary": final_state.get("summary") or "",
    }


def _last_ai_text(messages: List[Any]) -> str:
    from langchain_core.messages import AIMessage

    for m in reversed(messages):
        if isinstance(m, AIMessage):
            c = m.content
            if isinstance(c, str):
                return _clean_assistant_text(c)
            if isinstance(c, list):
                parts = []
                for p in c:
                    if isinstance(p, dict) and p.get("type") == "text":
                        parts.append(p.get("text", ""))
                return _clean_assistant_text("\n".join(p for p in parts if p))
    return ""


_SQL_FENCE_RE = None
_MD_TABLE_RE = None


def _clean_assistant_text(text: str) -> str:
    """Strip any SQL code fences and markdown tables the LLM may have echoed.

    The UI renders SQL and tables from structured fields in the response —
    keeping them in the prose reply duplicates the content and looks broken.
    """
    import re

    global _SQL_FENCE_RE, _MD_TABLE_RE
    if _SQL_FENCE_RE is None:
        _SQL_FENCE_RE = re.compile(r"```(?:sql)?\s*\n.*?\n```", re.IGNORECASE | re.DOTALL)
    if _MD_TABLE_RE is None:
        # Header row + separator (--- | --- | ...) + any number of body rows.
        _MD_TABLE_RE = re.compile(
            r"(?:^\s*\|?[^\n]*\|[^\n]*\n)"          # header
            r"(?:^\s*\|?[\s\-|:]+\|[\s\-|:]+\n)"    # separator with pipes
            r"(?:^\s*\|?[^\n]*\|[^\n]*\n?)+",       # body rows
            re.MULTILINE,
        )

    t = _SQL_FENCE_RE.sub("", text)
    t = _MD_TABLE_RE.sub("", t)
    # Also drop leftover headings like "SQL Genie produced:" / "Result:" on their own line.
    t = re.sub(
        r"^\s*(SQL Genie produced|Result|SQL)\s*:\s*\n+",
        "",
        t,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    # Collapse runs of blank lines.
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# --------------------------------------------------------------------
# Option D: reset Genie conv on a thread (keeps agent summary intact)
# --------------------------------------------------------------------


@app.post("/api/threads/{thread_id}/reset-genie")
async def reset_genie(
    thread_id: str,
    body: ResetGenieRequest = ResetGenieRequest(),
    ctx: _AuthCtx = Depends(get_current_ctx),
) -> Dict[str, Any]:
    """Rotate the sticky Genie conv id(s). With `space=None`, reset all
    configured spaces. With `space="sales"` or `"crm"`, reset just that one.
    Pre-existing tool calls for the reset space(s) are marked as cleared, so
    their SQL + table are hidden on replay.
    """
    owned = await lakebase.get_for_user(ctx.user.username, thread_id)
    if not owned:
        raise HTTPException(status_code=404, detail="thread not found")

    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = await graph.aget_state(config)
    state_values = snapshot.values if snapshot else {}
    prior_results = state_values.get("genie_results") or []
    conv_ids = dict(state_values.get("genie_conv_ids") or {})
    # Back-compat: fold the legacy single genie_conv_id into sales.
    if "sales" not in conv_ids and state_values.get("genie_conv_id"):
        conv_ids["sales"] = state_values["genie_conv_id"]

    all_spaces = list(settings.genie_spaces().keys())
    target_spaces = all_spaces if body.space is None else [body.space]
    if body.space is not None and body.space not in all_spaces:
        raise HTTPException(status_code=400, detail=f"unknown space: {body.space!r}")

    # Clear each targeted space's sticky conv id and collect tool_call_ids to
    # mark as cleared. Tool calls with no space field are treated as sales
    # (pre-multi-space history).
    cleared_ids: List[str] = []
    for r in prior_results:
        tc_id = r.get("tool_call_id")
        if not tc_id:
            continue
        space_of_result = r.get("space", "sales")
        if space_of_result in target_spaces:
            cleared_ids.append(tc_id)

    new_conv_ids = {k: v for k, v in conv_ids.items() if k not in target_spaces}

    update: Dict[str, Any] = {
        "genie_conv_ids": new_conv_ids,
        "genie_cleared_tool_call_ids": cleared_ids,
    }
    # Keep legacy field in sync if sales was reset.
    if "sales" in target_spaces:
        update["genie_conv_id"] = None
    await graph.aupdate_state(config, update)

    # Clear the sidebar pointer only if we reset sales (that column represents
    # the sales conv; a future extension would track all spaces on the row).
    if "sales" in target_spaces:
        await lakebase.set_genie_conv_id(thread_id, None)

    await flow_events.publish(
        ctx.session_id,
        step="agent_reset_genie",
        status="ok",
        title=f"Reset Genie: {', '.join(target_spaces) or 'none'}",
        detail=(
            f"Cleared data visuals on {len(cleared_ids)} prior turn(s); "
            "agent summary preserved."
        ),
        payload={
            "thread_id": thread_id,
            "spaces_reset": target_spaces,
            "cleared": len(cleared_ids),
        },
    )
    return {
        "ok": True,
        "thread_id": thread_id,
        "spaces_reset": target_spaces,
        "cleared": len(cleared_ids),
    }


# --------------------------------------------------------------------
# Threads list + transcript
# --------------------------------------------------------------------


@app.get("/api/conversations")
async def list_conversations(
    ctx: _AuthCtx = Depends(get_current_ctx),
) -> Dict[str, Any]:
    rows = await lakebase.list_for_user(ctx.user.username)
    # Normalize timestamps to iso strings for the UI.
    for r in rows:
        for k in ("created_at", "last_active_at"):
            v = r.get(k)
            if v is not None and not isinstance(v, str):
                r[k] = v.isoformat()
    return {"conversations": rows}


@app.get("/api/conversations/{thread_id}")
async def get_thread(
    thread_id: str, ctx: _AuthCtx = Depends(get_current_ctx)
) -> Dict[str, Any]:
    """Return the thread transcript + current summary from LangGraph state.

    Threads created before we started persisting `genie_results` in state will
    have tool calls with no attached sql/rows/columns. For those we fall back
    to Genie's own list-messages API using the thread's sticky `genie_conv_id`,
    and re-attach the structured data by message order.
    """
    owned = await lakebase.get_for_user(ctx.user.username, thread_id)
    if not owned:
        raise HTTPException(status_code=404, detail="thread not found")

    graph = get_graph()
    snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    state = snapshot.values if snapshot else {}

    messages_list = state.get("messages") or []
    genie_results = list(state.get("genie_results") or [])
    cleared_ids = set(state.get("genie_cleared_tool_call_ids") or [])

    # Fallback: if state is missing results for any tool call, hydrate from
    # Genie — but skip tool calls that have been cleared by a reset (they'll
    # render with a "cleared" marker, no point fetching).
    tool_call_ids = _all_tool_call_ids(messages_list)
    covered = {r.get("tool_call_id") for r in genie_results}
    missing = [
        tid for tid in tool_call_ids if tid not in covered and tid not in cleared_ids
    ]
    if missing and owned.get("genie_conv_id") and settings.genie_space_id:
        extra = await _hydrate_from_genie(
            genie_conv_id=owned["genie_conv_id"],
            sp_label=ctx.user.sp_label,
            session_id=ctx.session_id,
            tool_call_ids_needing_results=missing,
        )
        genie_results.extend(extra)

    bubbles = _bubbles_from_messages(messages_list, genie_results, cleared_ids)
    return {
        "thread_id": thread_id,
        "title": owned.get("title"),
        "genie_conv_id": owned.get("genie_conv_id"),
        "summary": state.get("summary") or "",
        "messages": bubbles,
    }


def _all_tool_call_ids(messages: List[Any]) -> List[str]:
    from langchain_core.messages import AIMessage

    ids: List[str] = []
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in getattr(m, "tool_calls", None) or []:
                tid = tc.get("id")
                if tid:
                    ids.append(tid)
    return ids


async def _hydrate_from_genie(
    *,
    genie_conv_id: str,
    sp_label: str,
    session_id: str,
    tool_call_ids_needing_results: List[str],
) -> List[Dict[str, Any]]:
    """Fetch Genie's message list + query results and re-key by the thread's
    tool_call_ids, in order. Genie doesn't know about our tool_call_ids, so we
    zip by position: the Nth Genie message with attachments fills the Nth
    missing tool_call_id.
    """
    from .genie_client import GenieClient, GenieError, extract_sql_and_text

    try:
        sp_token = await databricks_auth.get_sp_token(sp_label, session_id)
    except Exception:
        return []

    client = GenieClient(
        space_id=settings.genie_space_id,
        sp_token=sp_token,
        session_id=session_id,
    )
    try:
        raw_messages = await client.list_messages(genie_conv_id)
    except GenieError:
        return []

    raw_messages.sort(
        key=lambda m: m.get("created_timestamp") or m.get("createdTimestamp") or 0
    )

    # Pair by position: iterate Genie messages that carry assistant attachments,
    # skip the ones that are just user prompts.
    hydrated: List[Dict[str, Any]] = []
    pending_ids = list(tool_call_ids_needing_results)
    for m in raw_messages:
        if not pending_ids:
            break
        sql, answer_text = extract_sql_and_text(m)
        if not (sql or answer_text):
            continue
        msg_id = m.get("message_id") or m.get("id")
        rows: Optional[List[Dict[str, Any]]] = None
        columns: Optional[List[Dict[str, str]]] = None
        if sql and msg_id:
            try:
                qr = await client.get_query_result(genie_conv_id, msg_id)
                columns = qr.get("columns") or None
                rows = qr.get("rows") or None
            except GenieError:
                pass
        tc_id = pending_ids.pop(0)
        hydrated.append(
            {
                "tool_call_id": tc_id,
                "conversation_id": genie_conv_id,
                "message_id": msg_id,
                "answer_text": answer_text,
                "sql": sql,
                "rows": rows,
                "columns": columns,
            }
        )
    await client.close()
    return hydrated


def _bubbles_from_messages(
    messages: List[Any],
    genie_results: List[Dict[str, Any]],
    cleared_tool_call_ids: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """Collapse the raw LangGraph message list into chat bubbles for the UI.

    For each tool call, we re-attach the structured sql/rows/columns from
    `genie_results` to the assistant message that followed the tool return —
    so on replay the UI gets the same shape as the live /api/chat response.

    Tool calls in `cleared_tool_call_ids` (set by a Genie reset) render with
    a `cleared` flag instead of sql/rows/columns — the UI swaps in a short
    "cleared as part of reset" note.
    """
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    cleared_set = cleared_tool_call_ids or set()
    result_by_call_id = {
        r.get("tool_call_id"): r for r in genie_results if r.get("tool_call_id")
    }
    pending_tool_call_ids: List[str] = []
    out: List[Dict[str, Any]] = []

    for m in messages:
        if isinstance(m, SystemMessage):
            continue
        if isinstance(m, HumanMessage):
            out.append({"role": "user", "text": _msg_text(m)})
        elif isinstance(m, AIMessage):
            tool_calls = getattr(m, "tool_calls", None) or []
            text = _clean_assistant_text(_msg_text(m))
            if tool_calls:
                for tc in tool_calls:
                    if tc.get("id"):
                        pending_tool_call_ids.append(tc["id"])
                continue
            bubble: Dict[str, Any] = {"role": "assistant", "text": text}
            bubble_results: List[Dict[str, Any]] = []
            any_cleared = False
            # Drain every tool call that preceded this assistant message —
            # a single turn may have hit multiple Genie spaces.
            while pending_tool_call_ids:
                tc_id = pending_tool_call_ids.pop(0)
                if tc_id in cleared_set:
                    any_cleared = True
                    continue
                res = result_by_call_id.get(tc_id)
                if res:
                    bubble_results.append(
                        {
                            "space": res.get("space"),
                            "space_label": res.get("space_label"),
                            "sql": res.get("sql"),
                            "rows": res.get("rows"),
                            "columns": res.get("columns"),
                        }
                    )
            if bubble_results:
                bubble["results"] = bubble_results
            if any_cleared and not bubble_results:
                bubble["cleared"] = True
            out.append(bubble)
        elif isinstance(m, ToolMessage):
            continue
    return out


def _msg_text(m: Any) -> str:
    c = getattr(m, "content", None)
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for p in c:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
        return "\n".join(p for p in parts if p)
    return ""


# --------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------


@app.on_event("startup")
async def _on_startup() -> None:
    # MLflow first so graph-compile traces are captured.
    tracing.init()
    sps = []
    if settings.sp_northstar_client_id:
        sps.append(f"northstar ({settings.sp_northstar_dealership})")
    if settings.sp_sunrise_client_id:
        sps.append(f"sunrise ({settings.sp_sunrise_dealership})")
    log.info("Configured SPs: %s", ", ".join(sps) if sps else "(none)")
    log.info(
        "GENIE_SPACE_ID %s",
        "set" if settings.genie_space_id else "NOT SET (chat will 503)",
    )
    log.info("FM endpoint: %s", settings.fm_endpoint_name)
    log.info(
        "Summary thresholds: turns>%d OR tokens>%d, keep_last=%d",
        settings.summary_turn_threshold,
        settings.summary_token_threshold,
        settings.summary_keep_last,
    )

    if settings.lakebase_pg_uri:
        try:
            await lakebase.init()
        except Exception as e:
            log.error("Lakebase init failed: %s", e)
            log.error("Chat will 503 until LAKEBASE_PG_URI is a reachable instance.")
    else:
        log.warning("LAKEBASE_PG_URI not set; chat will 503.")


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    await databricks_auth.shutdown()
    await sql_client.shutdown()
    await lakebase.shutdown()
