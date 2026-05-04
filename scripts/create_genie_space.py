#!/usr/bin/env python
"""
Creates (or reuses) a Genie space over the seeded demo tables and grants
CAN_RUN to both service principals.

Run AFTER scripts/setup_databricks.py.

Because programmatic creation of a Genie space currently requires a
proto-serialized `GenieSpaceExport` blob (undocumented), this script
attempts the create call and, if it fails, prints clear instructions
for creating the space manually in the UI and re-running with:

    GENIE_SPACE_ID=<id> .venv/bin/python scripts/create_genie_space.py

On success, the script:
  - GRANTs CAN_RUN on the space to both SPs (via the permissions API).
  - Writes the `GENIE_SPACE_ID` value into `.env` so other agents pick it up.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env"
load_dotenv(ENV_FILE)

DBX_HOST = os.environ["DBX_HOST"].rstrip("/")
DBX_PROFILE = os.environ["DBX_PROFILE"]
DBX_WAREHOUSE_ID = os.environ["DBX_WAREHOUSE_ID"]
DBX_CATALOG = os.environ["DBX_CATALOG"]
DBX_SCHEMA = os.environ["DBX_SCHEMA"]

SP_NORTHSTAR_CLIENT_ID = os.environ["SP_NORTHSTAR_CLIENT_ID"]
SP_SUNRISE_CLIENT_ID = os.environ["SP_SUNRISE_CLIENT_ID"]

TABLES = ["dealerships", "vehicles", "sales", "service_tickets"]
SPACE_TITLE = "Dealership Analytics"
SPACE_DESCRIPTION = (
    "Ask questions about vehicle inventory, sales, and service tickets."
)


def log(msg: str, indent: int = 0) -> None:
    print("  " * indent + msg, flush=True)


def get_user_pat() -> str:
    """Use the user's CLI profile to get a PAT-like OAuth token."""
    res = subprocess.run(
        [
            "databricks",
            "auth",
            "token",
            "--host",
            DBX_HOST,
            "--profile",
            DBX_PROFILE,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        # Fall back to --host only
        res = subprocess.run(
            ["databricks", "auth", "token", "--host", DBX_HOST],
            capture_output=True,
            text=True,
            check=True,
        )
    return json.loads(res.stdout)["access_token"]


# --------------------------------------------------------------------------
# Space create / reuse
# --------------------------------------------------------------------------


def find_existing_space(client: httpx.Client) -> str | None:
    """Look for an existing space with our title (idempotent re-run)."""
    r = client.get(f"{DBX_HOST}/api/2.0/genie/spaces")
    if r.status_code != 200:
        return None
    for space in r.json().get("spaces", []):
        if space.get("title") == SPACE_TITLE:
            return space.get("space_id")
    return None


def attempt_programmatic_create(client: httpx.Client) -> str | None:
    """
    Try to POST /api/2.0/genie/spaces with a serialized payload.

    This public API currently requires a proto-serialized `GenieSpaceExport`
    blob whose schema is not publicly documented, so this is a best-effort
    attempt. Returns the space_id on success, else None.
    """
    candidates = []

    # Candidate 1: version + warehouse + dataset table refs (common
    # export-schema layout).
    candidates.append(
        {
            "warehouse_id": DBX_WAREHOUSE_ID,
            "title": SPACE_TITLE,
            "description": SPACE_DESCRIPTION,
            "serialized_space": json.dumps(
                {
                    "version": 2,
                    "display_name": SPACE_TITLE,
                    "description": SPACE_DESCRIPTION,
                    "warehouse_id": DBX_WAREHOUSE_ID,
                    "datasets": [
                        {
                            "table": {
                                "catalog": DBX_CATALOG,
                                "schema": DBX_SCHEMA,
                                "table": t,
                            }
                        }
                        for t in TABLES
                    ],
                }
            ),
        }
    )

    # Candidate 2: version 1
    candidates.append(
        {
            "warehouse_id": DBX_WAREHOUSE_ID,
            "title": SPACE_TITLE,
            "description": SPACE_DESCRIPTION,
            "serialized_space": json.dumps(
                {
                    "version": 1,
                    "title": SPACE_TITLE,
                    "description": SPACE_DESCRIPTION,
                    "tables": [
                        f"{DBX_CATALOG}.{DBX_SCHEMA}.{t}" for t in TABLES
                    ],
                }
            ),
        }
    )

    for body in candidates:
        r = client.post(f"{DBX_HOST}/api/2.0/genie/spaces", json=body)
        if r.status_code == 200:
            data = r.json()
            space_id = data.get("space_id") or data.get("id")
            if space_id:
                return space_id
        else:
            log(f"create attempt failed {r.status_code}: {r.text[:200]}", indent=1)
    return None


def print_manual_instructions() -> None:
    print()
    print("=" * 70)
    print("MANUAL STEP REQUIRED")
    print("=" * 70)
    print("The Genie space create REST API requires an undocumented proto blob,")
    print("so we could not create the space programmatically.")
    print()
    print("Please do this in the UI:")
    print(f"  1. Open {DBX_HOST}/genie")
    print("  2. Click 'New' -> 'Genie space'")
    print(f"  3. Title:       {SPACE_TITLE!r}")
    print(f"  4. Description: {SPACE_DESCRIPTION!r}")
    print(f"  5. Warehouse:   {DBX_WAREHOUSE_ID}")
    print("  6. Add the following tables as data sources:")
    for t in TABLES:
        print(f"       - {DBX_CATALOG}.{DBX_SCHEMA}.{t}")
    print("  7. Save and copy the space_id from the URL")
    print("     (looks like https://.../genie/rooms/<space_id>).")
    print()
    print("Then re-run this script with:")
    print(
        "   GENIE_SPACE_ID=<id> .venv/bin/python scripts/create_genie_space.py"
    )
    print(
        "(or paste it into .env under GENIE_SPACE_ID=... and re-run the script)"
    )
    print("=" * 70)


# --------------------------------------------------------------------------
# Grants via permissions API
# --------------------------------------------------------------------------


def grant_can_run(client: httpx.Client, space_id: str) -> None:
    """Grant CAN_RUN on the Genie space to both SPs."""
    url = f"{DBX_HOST}/api/2.0/permissions/genie/{space_id}"
    body = {
        "access_control_list": [
            {
                "service_principal_name": SP_NORTHSTAR_CLIENT_ID,
                "permission_level": "CAN_RUN",
            },
            {
                "service_principal_name": SP_SUNRISE_CLIENT_ID,
                "permission_level": "CAN_RUN",
            },
        ]
    }
    # PATCH adds without removing existing permissions.
    r = client.patch(url, json=body)
    if r.status_code not in (200, 204):
        raise RuntimeError(
            f"Failed to grant CAN_RUN: {r.status_code} {r.text[:400]}"
        )
    log(
        f"granted CAN_RUN to northstar ({SP_NORTHSTAR_CLIENT_ID[:8]}…) "
        f"and sunrise ({SP_SUNRISE_CLIENT_ID[:8]}…)",
        indent=1,
    )


# --------------------------------------------------------------------------
# .env update
# --------------------------------------------------------------------------


def update_env_file(space_id: str) -> None:
    text = ENV_FILE.read_text()
    new_line = f"GENIE_SPACE_ID={space_id}"
    updated_lines = []
    found = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("GENIE_SPACE_ID="):
            updated_lines.append(new_line)
            found = True
        else:
            updated_lines.append(line)
    if not found:
        updated_lines.append(new_line)
    ENV_FILE.write_text("\n".join(updated_lines) + "\n")
    log(f".env updated with GENIE_SPACE_ID={space_id}", indent=1)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    try:
        pat = get_user_pat()
        headers = {
            "Authorization": f"Bearer {pat}",
            "Content-Type": "application/json",
        }
        client = httpx.Client(headers=headers, timeout=60.0)

        log("Step 1) Locating Genie space...")
        # Precedence: explicit env var > existing space with our title > create.
        space_id = os.environ.get("GENIE_SPACE_ID") or None
        if space_id:
            log(f"using GENIE_SPACE_ID from env: {space_id}", indent=1)
        else:
            existing = find_existing_space(client)
            if existing:
                space_id = existing
                log(f"found existing space with matching title: {space_id}", indent=1)
            else:
                log("attempting programmatic create...", indent=1)
                space_id = attempt_programmatic_create(client)
                if space_id:
                    log(f"created space: {space_id}", indent=1)

        if not space_id:
            print_manual_instructions()
            return 2

        log("Step 2) Granting CAN_RUN to both SPs...")
        grant_can_run(client, space_id)

        log("Step 3) Updating .env...")
        update_env_file(space_id)

        print()
        print("=" * 60)
        print("GENIE SPACE READY")
        print("=" * 60)
        print(f"GENIE_SPACE_ID={space_id}")
        print(f"UI: {DBX_HOST}/genie/rooms/{space_id}")
        print()
        print("Paste the line below into .env (already done automatically):")
        print(f"  GENIE_SPACE_ID={space_id}")
        return 0
    except Exception as exc:
        print(f"\nFAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
