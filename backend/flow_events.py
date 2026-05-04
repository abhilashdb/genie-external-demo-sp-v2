"""In-memory SSE event bus keyed by session_id.

Each session_id owns an asyncio.Queue; publishers enqueue JSON-serializable
event dicts, subscribers yield SSE-formatted strings. Queues that haven't
been touched for > 1 hour are garbage-collected lazily on publish/subscribe.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, Optional

# session_id -> (Queue, last_accessed_monotonic_ts)
_queues: Dict[str, asyncio.Queue] = {}
_last_access: Dict[str, float] = {}

# Lock to guard dict mutations across concurrent tasks.
_lock = asyncio.Lock()

# Queues idle longer than this are GC'd.
_TTL_SECONDS = 60 * 60  # 1 hour

# Per-queue max size — if the subscriber is absent, events drop silently.
_MAX_QUEUE_SIZE = 1000


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _gc_expired_locked() -> None:
    """Remove queues that haven't been accessed within TTL. Caller holds _lock."""
    now = time.monotonic()
    expired = [sid for sid, ts in _last_access.items() if now - ts > _TTL_SECONDS]
    for sid in expired:
        _queues.pop(sid, None)
        _last_access.pop(sid, None)


async def _get_or_create_queue(session_id: str) -> asyncio.Queue:
    async with _lock:
        _gc_expired_locked()
        q = _queues.get(session_id)
        if q is None:
            q = asyncio.Queue(maxsize=_MAX_QUEUE_SIZE)
            _queues[session_id] = q
        _last_access[session_id] = time.monotonic()
        return q


async def publish(
    session_id: str,
    step: str,
    status: str,
    title: str,
    detail: str = "",
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    """Enqueue a flow event for the given session. No-op if session_id is falsy."""
    if not session_id:
        return
    event = {
        "ts": _now_iso(),
        "step": step,
        "status": status,
        "title": title,
        "detail": detail or "",
        "payload": payload or {},
    }
    q = await _get_or_create_queue(session_id)
    try:
        q.put_nowait(event)
    except asyncio.QueueFull:
        # Drop silently rather than blocking the caller on a stalled subscriber.
        pass


async def subscribe(session_id: str) -> AsyncGenerator[str, None]:
    """Yield SSE-formatted strings for a session until the consumer disconnects."""
    if not session_id:
        return
    q = await _get_or_create_queue(session_id)

    # Heartbeat keeps intermediaries from closing the connection.
    heartbeat_interval = 15.0

    while True:
        try:
            event = await asyncio.wait_for(q.get(), timeout=heartbeat_interval)
        except asyncio.TimeoutError:
            # Send a comment line as a heartbeat; clients ignore comments.
            yield ": keepalive\n\n"
            async with _lock:
                _last_access[session_id] = time.monotonic()
            continue

        async with _lock:
            _last_access[session_id] = time.monotonic()

        data = json.dumps(event, default=str)
        yield f"event: flow\ndata: {data}\n\n"
