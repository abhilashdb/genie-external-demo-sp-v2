"""Lakebase-backed persistence: LangGraph checkpointer + app_conversations table.

Owns a single psycopg AsyncConnectionPool pointed at the Lakebase Postgres URI.
Exposes:
  * get_checkpointer() -> AsyncPostgresSaver  (LangGraph state snapshots)
  * upsert_conversation / list_for_user / get_for_user / set_genie_conv_id

The checkpointer schema is auto-created via AsyncPostgresSaver.setup(); the
app_conversations table is created here.

Schema:
  app_conversations(
    thread_id       TEXT PRIMARY KEY,     -- "<local_user>:<uuid>"
    local_user      TEXT NOT NULL,
    sp_label        TEXT NOT NULL,
    genie_conv_id   TEXT,                 -- sticky per thread; may be NULL until first Genie call
    title           TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_active_at  TIMESTAMPTZ NOT NULL DEFAULT now()
  )
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

from psycopg_pool import AsyncConnectionPool

from .config import settings

log = logging.getLogger("genie_sp_demo_v2.lakebase")

_pool: Optional[AsyncConnectionPool] = None
_checkpointer = None  # lazily imported to avoid pulling LangGraph at cold import

_APP_CONV_DDL = """
CREATE TABLE IF NOT EXISTS app_conversations (
    thread_id       TEXT PRIMARY KEY,
    local_user      TEXT NOT NULL,
    sp_label        TEXT NOT NULL,
    genie_conv_id   TEXT,
    title           TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_active_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_app_conv_user_active
    ON app_conversations (local_user, last_active_at DESC);
"""


class LakebaseNotConfigured(RuntimeError):
    pass


def _require_uri() -> str:
    if not settings.lakebase_pg_uri:
        raise LakebaseNotConfigured(
            "LAKEBASE_PG_URI is not set — provision a Lakebase instance and "
            "update .env before hitting /api/chat."
        )
    return settings.lakebase_pg_uri


async def init() -> None:
    """Open the pool, bootstrap LangGraph checkpoint tables + app_conversations."""
    global _pool, _checkpointer
    if _pool is not None:
        return

    uri = _require_uri()
    _pool = AsyncConnectionPool(
        conninfo=uri,
        min_size=1,
        max_size=10,
        kwargs={"autocommit": True},
        open=False,
    )
    await _pool.open(wait=True, timeout=15)

    # Create our app table.
    async with _pool.connection() as conn:
        await conn.execute(_APP_CONV_DDL)

    # Create LangGraph checkpointer tables.
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    _checkpointer = AsyncPostgresSaver(_pool)
    await _checkpointer.setup()
    log.info("Lakebase ready: app_conversations + LangGraph checkpoint tables")


async def shutdown() -> None:
    global _pool, _checkpointer
    if _pool is not None:
        await _pool.close()
    _pool = None
    _checkpointer = None


def get_checkpointer():
    if _checkpointer is None:
        raise LakebaseNotConfigured("Lakebase not initialized; call init() first.")
    return _checkpointer


@asynccontextmanager
async def conn() -> AsyncIterator[Any]:
    if _pool is None:
        raise LakebaseNotConfigured("Lakebase not initialized; call init() first.")
    async with _pool.connection() as c:
        yield c


# ----- app_conversations CRUD -----


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _title_from(message: Optional[str], limit: int = 60) -> str:
    if not message:
        return "(untitled)"
    t = message.strip().splitlines()[0]
    if len(t) > limit:
        t = t[: limit - 1].rstrip() + "…"
    return t or "(untitled)"


async def upsert_conversation(
    *,
    thread_id: str,
    local_user: str,
    sp_label: str,
    first_message: Optional[str],
) -> None:
    """Insert on first turn; bump last_active_at on follow-ups."""
    async with conn() as c:
        await c.execute(
            """
            INSERT INTO app_conversations
                (thread_id, local_user, sp_label, title, created_at, last_active_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (thread_id) DO UPDATE
                SET last_active_at = EXCLUDED.last_active_at
            """,
            (
                thread_id,
                local_user,
                sp_label,
                _title_from(first_message),
                _now(),
                _now(),
            ),
        )


async def set_genie_conv_id(thread_id: str, genie_conv_id: Optional[str]) -> None:
    """Store the sticky Genie conv_id for this thread. Pass None to clear (Option D reset)."""
    async with conn() as c:
        await c.execute(
            "UPDATE app_conversations SET genie_conv_id = %s, last_active_at = %s "
            "WHERE thread_id = %s",
            (genie_conv_id, _now(), thread_id),
        )


async def list_for_user(local_user: str) -> List[Dict[str, Any]]:
    async with conn() as c:
        async with c.cursor() as cur:
            await cur.execute(
                """
                SELECT thread_id, title, genie_conv_id, created_at, last_active_at
                FROM app_conversations
                WHERE local_user = %s
                ORDER BY last_active_at DESC
                """,
                (local_user,),
            )
            rows = await cur.fetchall()
            cols = [d.name for d in cur.description]
    return [dict(zip(cols, r)) for r in rows]


async def get_for_user(local_user: str, thread_id: str) -> Optional[Dict[str, Any]]:
    async with conn() as c:
        async with c.cursor() as cur:
            await cur.execute(
                """
                SELECT thread_id, local_user, sp_label, title, genie_conv_id,
                       created_at, last_active_at
                FROM app_conversations
                WHERE local_user = %s AND thread_id = %s
                """,
                (local_user, thread_id),
            )
            row = await cur.fetchone()
            if row is None:
                return None
            cols = [d.name for d in cur.description]
    return dict(zip(cols, row))
