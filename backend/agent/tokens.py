"""Tiny token-count helper for summarization triggers.

Uses tiktoken's cl100k_base as a rough stand-in for Claude's tokenizer; we
only need an order-of-magnitude count for compression thresholds.
"""

from __future__ import annotations

from typing import Any, Iterable

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover — fallback for environments without tiktoken
    _enc = None


def _text_of(m: Any) -> str:
    if isinstance(m, dict):
        c = m.get("content")
    else:
        c = getattr(m, "content", None)
    if c is None:
        return ""
    if isinstance(c, list):
        # LC content blocks — flatten text parts only.
        parts = []
        for p in c:
            if isinstance(p, dict):
                if p.get("type") == "text":
                    parts.append(p.get("text", ""))
            else:
                parts.append(str(p))
        return "\n".join(parts)
    return str(c)


def count(messages: Iterable[Any]) -> int:
    total = 0
    for m in messages:
        s = _text_of(m)
        if not s:
            continue
        if _enc is None:
            total += max(1, len(s) // 4)  # ~4 chars/token rough fallback
        else:
            total += len(_enc.encode(s))
    return total
