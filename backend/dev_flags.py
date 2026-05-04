"""Per-session dev toggles for demoing failure modes without real outages.

Currently supports rate-limit simulation: `arm_rate_limit(session_id, n)` makes
the next N Genie HTTP attempts for that session return a synthetic 429, so the
retry UX in the transparency pane is visible even when the real backend has
headroom.
"""
from __future__ import annotations

import threading
from typing import Dict, Optional

_lock = threading.Lock()
# session_id -> {"rate_limit_remaining": int, "rate_limit_status": int}
_state: Dict[str, Dict[str, int]] = {}


def arm_rate_limit(session_id: str, count: int, status: int = 429) -> int:
    """Queue up `count` synthetic rate-limit responses for this session.

    Returns the new pending count. Max 10, min 0.
    """
    count = max(0, min(10, count))
    with _lock:
        _state[session_id] = {"rate_limit_remaining": count, "rate_limit_status": status}
    return count


def consume_rate_limit(session_id: str) -> Optional[int]:
    """If the session has pending simulated rate limits, decrement and return
    the status code to use. Otherwise return None.
    """
    with _lock:
        s = _state.get(session_id)
        if not s or s.get("rate_limit_remaining", 0) <= 0:
            return None
        s["rate_limit_remaining"] -= 1
        return int(s.get("rate_limit_status") or 429)


def peek_rate_limit(session_id: str) -> int:
    with _lock:
        return int((_state.get(session_id) or {}).get("rate_limit_remaining", 0))


def clear(session_id: str) -> None:
    with _lock:
        _state.pop(session_id, None)
