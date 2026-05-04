"""Per-request context for secrets and scratch values.

Anything we don't want to leak into MLflow traces (SP tokens, raw credentials)
lives here — NOT in LangGraph state. Set at the /api/chat boundary, read from
inside graph nodes.

asyncio ContextVars propagate across `await` boundaries within the same task,
which is exactly the lifetime of one `graph.ainvoke(...)` call.
"""

from __future__ import annotations

from contextvars import ContextVar

_sp_token: ContextVar[str] = ContextVar("sp_token", default="")
_session_id: ContextVar[str] = ContextVar("session_id", default="")


def set_sp_token(token: str) -> None:
    _sp_token.set(token or "")


def get_sp_token() -> str:
    return _sp_token.get()


def set_session_id(sid: str) -> None:
    _session_id.set(sid or "")


def get_session_id() -> str:
    return _session_id.get()
