"""Debug helper: prints the raw Genie query-result payload shape."""
from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

HOST = os.environ["DBX_HOST"].rstrip("/")
SPACE_ID = os.environ["GENIE_SPACE_ID"]
CLIENT_ID = os.environ["SP_NORTHSTAR_CLIENT_ID"]
SECRET = os.environ["SP_NORTHSTAR_SECRET"]


async def get_token() -> str:
    basic = base64.b64encode(f"{CLIENT_ID}:{SECRET}".encode()).decode()
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(
            f"{HOST}/oidc/v1/token",
            headers={"Authorization": f"Basic {basic}"},
            data={"grant_type": "client_credentials", "scope": "all-apis"},
        )
        r.raise_for_status()
        return r.json()["access_token"]


async def main(query: str) -> None:
    token = await get_token()
    h = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(timeout=60, headers=h) as c:
        # start conversation
        r = await c.post(
            f"{HOST}/api/2.0/genie/spaces/{SPACE_ID}/start-conversation",
            json={"content": query},
        )
        r.raise_for_status()
        d = r.json()
        conv = d.get("conversation_id") or d["conversation"]["id"]
        msg = d.get("message_id") or d["message"]["id"]
        print(f"conv={conv} msg={msg}")

        # poll until done
        url = f"{HOST}/api/2.0/genie/spaces/{SPACE_ID}/conversations/{conv}/messages/{msg}"
        status = ""
        while True:
            r = await c.get(url)
            m = r.json()
            status = (m.get("status") or "").upper()
            print(f"  status={status}")
            if status in {"COMPLETED", "FAILED", "CANCELLED"}:
                break
            await asyncio.sleep(1.5)

        print("\n===== MESSAGE JSON =====")
        print(json.dumps(m, indent=2, default=str)[:4000])

        # fetch query-result
        r = await c.get(url + "/query-result")
        print(f"\n===== QUERY-RESULT HTTP {r.status_code} =====")
        print(json.dumps(r.json(), indent=2, default=str)[:6000])


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "Show my top 3 sales by sale_price"
    asyncio.run(main(q))
