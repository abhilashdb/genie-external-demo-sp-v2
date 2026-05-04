"""Async client wrapping the Databricks Genie Conversations REST API.

Endpoints (per API_CONTRACT.md):
  POST   /api/2.0/genie/spaces/{space_id}/start-conversation
  POST   /api/2.0/genie/spaces/{space_id}/conversations/{conv_id}/messages
  GET    /api/2.0/genie/spaces/{space_id}/conversations/{conv_id}/messages/{msg_id}
  GET    /api/2.0/genie/spaces/{space_id}/conversations/{conv_id}/messages/{msg_id}/query-result
  GET    /api/2.0/genie/spaces/{space_id}/conversations/{conv_id}/messages

Each HTTP call publishes a `genie_call` flow event. SQL (if any) is published
as a `genie_sql` event.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any, Dict, List, Optional, Tuple

import httpx

from . import dev_flags, flow_events
from .config import settings

_TERMINAL_STATUSES = {"COMPLETED", "FAILED", "CANCELLED", "QUERY_RESULT_EXPIRED"}

# Rate-limit retry config
_RATE_LIMIT_MAX_RETRIES = 5
_RATE_LIMIT_BASE_DELAY = 1.0    # seconds
_RATE_LIMIT_MAX_DELAY = 30.0    # cap a single backoff at 30s
_RATE_LIMIT_STATUSES = {429, 503}


class GenieError(RuntimeError):
    pass


class GenieClient:
    def __init__(
        self,
        space_id: str,
        sp_token: str,
        session_id: str,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        if not space_id:
            raise GenieError(
                "GENIE_SPACE_ID is empty — set it in .env after Agent A creates the space."
            )
        self.space_id = space_id
        self.sp_token = sp_token
        self.session_id = session_id
        self._owned_client = http_client is None
        self._client = http_client or httpx.AsyncClient(timeout=30.0)
        self._base = settings.api_base()

    # ----- lifecycle -----

    async def close(self) -> None:
        if self._owned_client:
            await self._client.aclose()

    async def __aenter__(self) -> "GenieClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    # ----- low level -----

    def _auth_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.sp_token}",
            "Content-Type": "application/json",
        }

    def _url(self, path: str) -> str:
        return f"{self._base}{path}"

    async def _request(
        self, method: str, path: str, json_body: Optional[Dict[str, Any]] = None
    ) -> httpx.Response:
        """Issue a request with transparent rate-limit retries.

        On HTTP 429/503, we honor Retry-After if present, otherwise backoff
        exponentially with jitter. Each retry emits a `genie_retry` flow event
        so the UI can show the retry attempt and delay.
        """
        url = self._url(path)
        attempts = 0
        last_exc: Optional[Exception] = None

        while True:
            # Dev toggle: simulate a rate-limit response if armed for this session.
            # This uses the exact same retry/backoff code path as a real 429, so
            # the transparency pane shows identical events (plus a `simulated: true`
            # marker in the payload).
            if attempts < _RATE_LIMIT_MAX_RETRIES:
                sim_status = dev_flags.consume_rate_limit(self.session_id)
                if sim_status:
                    attempts += 1
                    delay = _compute_backoff(None, attempts)
                    await flow_events.publish(
                        self.session_id,
                        step="genie_rate_limit",
                        status="pending",
                        title=f"[simulated] Rate limited (HTTP {sim_status}) — retrying",
                        detail=f"attempt {attempts}/{_RATE_LIMIT_MAX_RETRIES}, backing off {delay:.1f}s",
                        payload={
                            "endpoint": path,
                            "method": method,
                            "http_status": sim_status,
                            "attempt": attempts,
                            "max_retries": _RATE_LIMIT_MAX_RETRIES,
                            "delay_seconds": round(delay, 2),
                            "retry_after_header": None,
                            "simulated": True,
                        },
                    )
                    await asyncio.sleep(delay)
                    await flow_events.publish(
                        self.session_id,
                        step="genie_retry",
                        status="pending",
                        title=f"[simulated] Retrying {method} {path} (attempt {attempts + 1})",
                        detail=f"waited {delay:.1f}s",
                        payload={
                            "endpoint": path,
                            "method": method,
                            "attempt": attempts + 1,
                            "simulated": True,
                        },
                    )
                    continue

            try:
                resp = await self._client.request(
                    method, url, headers=self._auth_headers(), json=json_body
                )
            except httpx.HTTPError as e:
                await flow_events.publish(
                    self.session_id,
                    step="genie_call",
                    status="error",
                    title=f"{method} {path} failed",
                    detail=str(e),
                    payload={"endpoint": path, "method": method, "error": str(e)},
                )
                raise GenieError(f"Genie {method} {path} failed: {e}") from e

            # Rate-limited? Retry with backoff, up to max attempts.
            if resp.status_code in _RATE_LIMIT_STATUSES and attempts < _RATE_LIMIT_MAX_RETRIES:
                attempts += 1
                delay = _compute_backoff(
                    resp.headers.get("Retry-After"), attempts
                )
                await flow_events.publish(
                    self.session_id,
                    step="genie_rate_limit",
                    status="pending",
                    title=f"Rate limited (HTTP {resp.status_code}) — retrying",
                    detail=f"attempt {attempts}/{_RATE_LIMIT_MAX_RETRIES}, backing off {delay:.1f}s",
                    payload={
                        "endpoint": path,
                        "method": method,
                        "http_status": resp.status_code,
                        "attempt": attempts,
                        "max_retries": _RATE_LIMIT_MAX_RETRIES,
                        "delay_seconds": round(delay, 2),
                        "retry_after_header": resp.headers.get("Retry-After"),
                    },
                )
                await asyncio.sleep(delay)
                await flow_events.publish(
                    self.session_id,
                    step="genie_retry",
                    status="pending",
                    title=f"Retrying {method} {path} (attempt {attempts + 1})",
                    detail=f"waited {delay:.1f}s",
                    payload={"endpoint": path, "method": method, "attempt": attempts + 1},
                )
                continue

            # Normal path: log the call result.
            status_tag = "ok" if 200 <= resp.status_code < 300 else "error"
            await flow_events.publish(
                self.session_id,
                step="genie_call",
                status=status_tag,
                title=f"{method} {path} -> {resp.status_code}"
                + (f" (after {attempts} retries)" if attempts else ""),
                detail="",
                payload={
                    "endpoint": path,
                    "method": method,
                    "http_status": resp.status_code,
                    "retries": attempts,
                },
            )
            if status_tag == "error":
                raise GenieError(
                    f"Genie {method} {path} returned HTTP {resp.status_code}: {resp.text[:300]}"
                )
            return resp

    # ----- high-level methods -----

    async def start_conversation(self, content: str) -> Tuple[str, str]:
        """POST start-conversation. Returns (conversation_id, message_id)."""
        path = f"/api/2.0/genie/spaces/{self.space_id}/start-conversation"
        resp = await self._request("POST", path, {"content": content})
        data = resp.json()
        conv_id = (
            data.get("conversation_id")
            or (data.get("conversation") or {}).get("id")
            or (data.get("conversation") or {}).get("conversation_id")
        )
        msg_id = (
            data.get("message_id")
            or (data.get("message") or {}).get("id")
            or (data.get("message") or {}).get("message_id")
        )
        if not conv_id or not msg_id:
            raise GenieError(
                f"start-conversation response missing conversation_id/message_id: {data}"
            )
        return conv_id, msg_id

    async def send_message(self, conv_id: str, content: str) -> str:
        """POST a follow-up message to an existing conversation. Returns message_id."""
        path = f"/api/2.0/genie/spaces/{self.space_id}/conversations/{conv_id}/messages"
        resp = await self._request("POST", path, {"content": content})
        data = resp.json()
        msg_id = data.get("message_id") or data.get("id")
        if not msg_id:
            raise GenieError(f"send_message response missing message_id: {data}")
        return msg_id

    async def poll_message(
        self, conv_id: str, msg_id: str, timeout_s: float = 60.0
    ) -> Dict[str, Any]:
        """Poll the message until it reaches a terminal status or we hit timeout.

        Exponential backoff: 0.5s -> 4s max.
        """
        path = f"/api/2.0/genie/spaces/{self.space_id}/conversations/{conv_id}/messages/{msg_id}"
        deadline = asyncio.get_event_loop().time() + timeout_s
        delay = 0.5
        last: Dict[str, Any] = {}
        while True:
            resp = await self._request("GET", path)
            last = resp.json()
            status = (last.get("status") or "").upper()
            if status in _TERMINAL_STATUSES:
                if status != "COMPLETED":
                    err = last.get("error") or {}
                    msg = err.get("message") if isinstance(err, dict) else str(err)
                    raise GenieError(
                        f"Genie message {msg_id} ended with status={status}: {msg or last}"
                    )
                return last
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise GenieError(
                    f"Timed out polling Genie message {msg_id} (last status={status or 'UNKNOWN'})"
                )
            await asyncio.sleep(min(delay, remaining))
            delay = min(delay * 1.6, 4.0)

    async def get_query_result(
        self, conv_id: str, msg_id: str
    ) -> Dict[str, Any]:
        """Fetch the tabular result of a Genie SQL attachment."""
        path = (
            f"/api/2.0/genie/spaces/{self.space_id}/conversations/{conv_id}"
            f"/messages/{msg_id}/query-result"
        )
        resp = await self._request("GET", path)
        raw = resp.json()
        columns, rows = _parse_query_result(raw)
        return {"columns": columns, "rows": rows, "raw": raw}

    async def list_messages(self, conv_id: str) -> List[Dict[str, Any]]:
        """Fetch all messages in a conversation (for history replay).

        Response shape varies; we accept either `{"messages": [...]}` or a bare
        list. Each message is returned as-is and normalized by the caller.
        """
        path = f"/api/2.0/genie/spaces/{self.space_id}/conversations/{conv_id}/messages"
        resp = await self._request("GET", path)
        data = resp.json()
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            msgs = data.get("messages") or data.get("data") or []
            if isinstance(msgs, list):
                return msgs
        return []

    # ----- helpers that aren't HTTP calls -----

    async def publish_sql(self, sql: str) -> None:
        if not sql:
            return
        await flow_events.publish(
            self.session_id,
            step="genie_sql",
            status="ok",
            title="Genie produced SQL",
            detail=sql,
            payload={"sql": sql},
        )


# ----- module-level helpers -----


def _compute_backoff(retry_after_header: Optional[str], attempt: int) -> float:
    """Decide how long to wait before the next retry.

    Honors Retry-After (seconds) if present; otherwise exponential backoff
    with full jitter (base * 2^(attempt-1), capped, plus ±25% jitter).
    """
    if retry_after_header:
        try:
            return max(0.1, min(float(retry_after_header), _RATE_LIMIT_MAX_DELAY))
        except ValueError:
            pass  # HTTP-date form — fall through to exponential
    base = _RATE_LIMIT_BASE_DELAY * (2 ** max(0, attempt - 1))
    capped = min(base, _RATE_LIMIT_MAX_DELAY)
    jitter = capped * 0.25 * (random.random() * 2 - 1)  # ±25%
    return max(0.1, capped + jitter)


# ----- module-level parsers (kept out of the class for easy unit testing) -----

def extract_sql_and_text(message_json: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Pull a `query` SQL string and a plain-text answer out of a Genie message.

    Genie messages carry a list of `attachments` where each attachment has
    one of `{"query": {...}, "text": {...}}`. We return the first query SQL
    we find, and concatenate text attachments into answer_text.
    """
    sql: Optional[str] = None
    text_chunks: List[str] = []

    attachments = message_json.get("attachments") or []
    for att in attachments:
        if not isinstance(att, dict):
            continue
        q = att.get("query")
        if isinstance(q, dict) and sql is None:
            # Common shapes: {"query": "SELECT ..."} or {"query": {"query": "..."}}
            candidate = q.get("query") or q.get("sql") or q.get("statement")
            if isinstance(candidate, str) and candidate.strip():
                sql = candidate.strip()
        t = att.get("text")
        if isinstance(t, dict):
            chunk = t.get("content") or t.get("text") or ""
            if isinstance(chunk, str) and chunk.strip():
                text_chunks.append(chunk.strip())

    # Some payloads put a top-level "content" or "response" field.
    if not text_chunks:
        top_text = message_json.get("content") or message_json.get("response")
        if isinstance(top_text, str) and top_text.strip():
            text_chunks.append(top_text.strip())

    answer = "\n\n".join(text_chunks) if text_chunks else None
    return sql, answer


def _parse_query_result(raw: Dict[str, Any]) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    """Convert a Genie query-result payload into (columns, rows[dict])."""
    # Shape varies; normalize both layouts we've seen:
    #   { "statement_response": { "manifest": {...}, "result": {...} } }
    #   { "manifest": {...}, "result": {...} }
    stmt = raw.get("statement_response") or raw
    manifest = stmt.get("manifest") or {}
    result = stmt.get("result") or {}

    schema = manifest.get("schema") or {}
    col_defs = schema.get("columns") or []
    columns: List[Dict[str, str]] = []
    col_names: List[str] = []
    for c in col_defs:
        name = c.get("name") or c.get("column_name") or ""
        col_type = c.get("type_text") or c.get("type_name") or c.get("type") or "STRING"
        columns.append({"name": name, "type": col_type})
        col_names.append(name)

    # Genie query-result uses format=PROTOBUF_ARRAY with result.data_typed_array.
    # Each element looks like {"values": [{"str": "..."}, {"str": "..."}, ...]}.
    # Older / SQL Statement Execution API returns result.data_array as list-of-lists.
    rows: List[Dict[str, Any]] = []

    typed = result.get("data_typed_array") or []
    for row in typed:
        values = (row or {}).get("values") or []
        row_dict: Dict[str, Any] = {}
        for i, v in enumerate(values):
            key = col_names[i] if i < len(col_names) else f"col_{i}"
            row_dict[key] = _unwrap_typed_value(v)
        rows.append(row_dict)

    if not rows:
        data_array = result.get("data_array") or result.get("data") or []
        for raw_row in data_array:
            if isinstance(raw_row, list):
                rows.append(
                    {
                        (col_names[i] if i < len(col_names) else f"col_{i}"): raw_row[i]
                        for i in range(len(raw_row))
                    }
                )
            elif isinstance(raw_row, dict):
                rows.append(raw_row)

    return columns, rows


def normalize_history(raw_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert Genie list-messages output into bubble records.

    Each raw Genie message pairs a user prompt with zero or more assistant
    attachments. We emit one `user` bubble followed (when content exists) by
    one `assistant` bubble so the UI can render the thread in chronological
    order. We deliberately skip fetching query-result rows for historical
    messages — that would be N extra round-trips; users can rerun the turn if
    they need the data.
    """
    def _ts(m: Dict[str, Any]) -> Any:
        return m.get("created_timestamp") or m.get("createdTimestamp") or 0

    bubbles: List[Dict[str, Any]] = []
    for m in sorted(raw_messages, key=_ts):
        if not isinstance(m, dict):
            continue
        prompt = m.get("content")
        if isinstance(prompt, str) and prompt.strip():
            bubbles.append({"role": "user", "text": prompt.strip()})
        sql, text = extract_sql_and_text(m)
        if text or sql:
            bubbles.append({
                "role": "assistant",
                "text": text,
                "sql": sql,
                "message_id": m.get("message_id") or m.get("id"),
            })
    return bubbles


def _unwrap_typed_value(v: Any) -> Any:
    """Flatten a Genie typed value cell to a primitive.

    Values arrive wrapped as `{"str": "..."}`, `{"bool": true}`, `{"long": "42"}`,
    `{"double": 1.5}`, `{"null": true}`, etc. We return the inner value or None.
    """
    if v is None:
        return None
    if not isinstance(v, dict):
        return v
    if v.get("null") is True or "null" in v and v.get("null"):
        return None
    for key in ("str", "string", "bool", "boolean", "long", "int", "integer",
                "double", "float", "decimal", "date", "timestamp", "value"):
        if key in v and v[key] is not None:
            return v[key]
    return v
