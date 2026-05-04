"""Thin async wrapper around the Databricks SQL Statements API.

Used by the backend for verification pings ("as SP X, SELECT COUNT(*) FROM sales")
to demonstrate that row-level security is applied per SP.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from . import flow_events
from .config import settings

_client: Optional[httpx.AsyncClient] = None


async def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=60.0)
    return _client


def _parse_columns_and_rows(
    body: Dict[str, Any]
) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    manifest = body.get("manifest") or {}
    result = body.get("result") or {}
    schema = manifest.get("schema") or {}
    col_defs = schema.get("columns") or []
    columns: List[Dict[str, str]] = []
    names: List[str] = []
    for c in col_defs:
        name = c.get("name") or c.get("column_name") or ""
        col_type = c.get("type_text") or c.get("type_name") or c.get("type") or "STRING"
        columns.append({"name": name, "type": col_type})
        names.append(name)
    data_array = result.get("data_array") or result.get("data") or []
    rows: List[Dict[str, Any]] = []
    for r in data_array:
        if isinstance(r, list):
            rows.append({
                names[i] if i < len(names) else f"col_{i}": r[i]
                for i in range(len(r))
            })
        elif isinstance(r, dict):
            rows.append(r)
    return columns, rows


async def execute_sql(
    sp_token: str, statement: str, session_id: str
) -> Dict[str, Any]:
    """Run `statement` on the configured warehouse using the SP's bearer token.

    Returns `{rows, columns, elapsed_ms, state}`. Publishes a `sql_execute` flow event.
    """
    endpoint = f"{settings.api_base()}/api/2.0/sql/statements"
    payload = {
        "warehouse_id": settings.dbx_warehouse_id,
        "statement": statement,
        "wait_timeout": "30s",
    }
    await flow_events.publish(
        session_id,
        step="sql_execute",
        status="pending",
        title="Executing SQL statement",
        detail=(statement[:200] + ("…" if len(statement) > 200 else "")),
        payload={
            "warehouse_id": settings.dbx_warehouse_id,
            "endpoint": "/api/2.0/sql/statements",
            "statement_preview": statement[:200],
        },
    )

    client = await _get_client()
    started = time.monotonic()
    try:
        resp = await client.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {sp_token}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
    except httpx.HTTPError as e:
        await flow_events.publish(
            session_id,
            step="error",
            status="error",
            title="SQL execute request failed",
            detail=str(e),
            payload={"where": "sql_execute", "message": str(e)},
        )
        raise

    elapsed_ms = int((time.monotonic() - started) * 1000)

    if resp.status_code >= 400:
        await flow_events.publish(
            session_id,
            step="sql_execute",
            status="error",
            title=f"SQL execute HTTP {resp.status_code}",
            detail=resp.text[:300],
            payload={
                "warehouse_id": settings.dbx_warehouse_id,
                "http_status": resp.status_code,
                "elapsed_ms": elapsed_ms,
            },
        )
        raise RuntimeError(
            f"SQL execute failed: HTTP {resp.status_code} — {resp.text[:300]}"
        )

    body = resp.json()
    state = ((body.get("status") or {}).get("state")) or body.get("state") or ""
    columns, rows = _parse_columns_and_rows(body)

    await flow_events.publish(
        session_id,
        step="sql_execute",
        status="ok",
        title=f"SQL execute {state or 'OK'}",
        detail=f"{len(rows)} rows in {elapsed_ms} ms",
        payload={
            "warehouse_id": settings.dbx_warehouse_id,
            "row_count": len(rows),
            "elapsed_ms": elapsed_ms,
        },
    )

    return {
        "rows": rows,
        "columns": columns,
        "elapsed_ms": elapsed_ms,
        "state": state,
    }


async def shutdown() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
