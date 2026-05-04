"""OAuth M2M token exchange for service principals.

Tokens are cached per sp_label until 60s before expiry. On cache miss we
POST client_credentials to `{DBX_HOST}/oidc/v1/token` with basic auth and
emit a `token_exchange` flow event with a redacted preview.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Dict, Optional

import httpx

from . import flow_events
from .config import settings
from .sp_mapping import resolve_sp

_SAFETY_MARGIN_S = 60.0


@dataclass
class _CachedToken:
    access_token: str
    expires_at_monotonic: float  # monotonic clock time when the token should be treated as expired


_cache: Dict[str, _CachedToken] = {}
_cache_lock = asyncio.Lock()

# Reused HTTP client for all token exchanges.
_client: Optional[httpx.AsyncClient] = None
_client_lock = asyncio.Lock()


async def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        async with _client_lock:
            if _client is None:
                _client = httpx.AsyncClient(timeout=30.0)
    return _client


def _token_preview(token: str) -> str:
    if not token:
        return ""
    return token[:12] + "…"  # first 12 chars + ellipsis


async def get_sp_token(sp_label: str, session_id: str) -> str:
    """Return a valid access_token for the given sp_label, using cache when possible.

    Emits a `token_exchange` flow event on every call (cache hit or miss).
    """
    client_id, client_secret, dealership, _app_id = resolve_sp(sp_label)
    now = time.monotonic()

    # Fast path: cache hit.
    async with _cache_lock:
        cached = _cache.get(sp_label)
        if cached is not None and cached.expires_at_monotonic - _SAFETY_MARGIN_S > now:
            remaining = int(cached.expires_at_monotonic - now)
            await flow_events.publish(
                session_id,
                step="token_exchange",
                status="ok",
                title=f"Reused cached token for {dealership}",
                detail=f"Valid for ~{remaining}s more",
                payload={
                    "endpoint": settings.token_url(),
                    "scope": "all-apis",
                    "expires_in": remaining,
                    "token_preview": _token_preview(cached.access_token),
                    "cache": "hit",
                },
            )
            return cached.access_token

    # Cache miss → exchange.
    endpoint = settings.token_url()
    await flow_events.publish(
        session_id,
        step="token_exchange",
        status="pending",
        title=f"Exchanging M2M credentials for {dealership}",
        detail=f"POST {endpoint}",
        payload={
            "endpoint": endpoint,
            "scope": "all-apis",
            "cache": "miss",
        },
    )

    client = await _get_client()
    try:
        resp = await client.post(
            endpoint,
            data={"grant_type": "client_credentials", "scope": "all-apis"},
            auth=(client_id, client_secret),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    except httpx.HTTPError as e:
        await flow_events.publish(
            session_id,
            step="error",
            status="error",
            title="Token exchange failed",
            detail=str(e),
            payload={"where": "token_exchange", "message": str(e)},
        )
        raise

    if resp.status_code != 200:
        body_preview = resp.text[:300]
        await flow_events.publish(
            session_id,
            step="error",
            status="error",
            title=f"Token exchange returned HTTP {resp.status_code}",
            detail=body_preview,
            payload={
                "where": "token_exchange",
                "message": body_preview,
                "http_status": resp.status_code,
            },
        )
        raise RuntimeError(
            f"Token exchange failed: HTTP {resp.status_code} — {body_preview}"
        )

    data = resp.json()
    access_token = data.get("access_token")
    expires_in = int(data.get("expires_in", 3600))
    if not access_token:
        raise RuntimeError("Token exchange response missing access_token")

    expires_at_monotonic = time.monotonic() + expires_in
    async with _cache_lock:
        _cache[sp_label] = _CachedToken(
            access_token=access_token, expires_at_monotonic=expires_at_monotonic
        )

    await flow_events.publish(
        session_id,
        step="token_exchange",
        status="ok",
        title=f"Issued M2M token for {dealership}",
        detail=f"expires_in={expires_in}s",
        payload={
            "endpoint": endpoint,
            "scope": "all-apis",
            "expires_in": expires_in,
            "token_preview": _token_preview(access_token),
            "cache": "miss",
        },
    )
    return access_token


async def shutdown() -> None:
    """Close the reused HTTP client on app shutdown."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
