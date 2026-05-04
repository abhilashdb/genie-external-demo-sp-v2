#!/usr/bin/env python
"""
Creates (or reuses) a CRM-focused Genie space over the abhilash_r.genie_crm
tables and grants CAN_RUN to both service principals.

Run AFTER scripts/setup_crm_databricks.py.

Sibling script to create_genie_space.py. Same API-call pattern; different
title / tables / sample questions. Does NOT write to .env — instead prints
`GENIE_SPACE_CRM_ID=<id>` and a scope description so the user can paste them.

Run with:  uv run python scripts/create_crm_genie_space.py
           (or .venv/bin/python scripts/create_crm_genie_space.py)

Override via env:
    GENIE_SPACE_CRM_ID=<id>  — skip discovery/create, just grant/verify.
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
CRM_SCHEMA = "genie_crm"

SP_NORTHSTAR_CLIENT_ID = os.environ["SP_NORTHSTAR_CLIENT_ID"]
SP_SUNRISE_CLIENT_ID = os.environ["SP_SUNRISE_CLIENT_ID"]

TABLES = ["leads", "opportunities", "activities", "sales_targets", "forecasts"]
SPACE_TITLE = "Dealership CRM — Leads, Pipeline, Forecasts"
SPACE_DESCRIPTION = (
    "CRM / pipeline analytics for dealership sales teams. Ask about leads, "
    "open opportunities, salesperson pipeline, quota attainment, and revenue "
    "forecasts. NOT for completed-sale or service-ticket history — those "
    "questions belong to the sales-and-service Genie space."
)
SAMPLE_QUESTIONS = [
    "Show my open opportunities by stage.",
    "Which salespeople are above target this month?",
    "What's the forecast for next month?",
    "Which leads haven't had activity in 30+ days?",
    "What's the pipeline value by vehicle make?",
]

# One-paragraph scope suitable for GENIE_SPACE_CRM_SCOPE env var. Mirrors
# the shape of the existing GENIE_SPACE_SCOPE value in .env.
SCOPE_DESCRIPTION = (
    "In scope: CRM / sales pipeline data for dealerships — leads (source, "
    "stage, assigned salesperson, recent activity), opportunities (stage, "
    "expected close date, expected value, vehicle make/model), activities "
    "(calls, emails, demos, test drives, quotes), salesperson targets vs. "
    "achievement by month, and per-dealership monthly revenue/unit forecasts "
    "with confidence. Out of scope: completed vehicle sales history, service "
    "tickets, inventory, finance/accounting, HR, and marketing-campaign "
    "analytics — those belong to the sales-and-service Genie space."
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
    """
    Look for an existing space with our title (idempotent re-run). The
    authoritative list-all endpoint is /api/2.0/genie/spaces; we match on
    the `title` field there.
    """
    r = client.get(f"{DBX_HOST}/api/2.0/genie/spaces")
    if r.status_code != 200:
        return None
    for space in r.json().get("spaces", []):
        if space.get("title") == SPACE_TITLE:
            return space.get("space_id")
    return None


def create_space(client: httpx.Client) -> str | None:
    """
    Create a new Genie space via the underlying data-rooms API
    (POST /api/2.0/data-rooms). The `genie/spaces` endpoint requires a
    proto-serialized blob, but data-rooms accepts a plain JSON shape with
    `display_name`, `description`, `warehouse_id`, `table_identifiers`,
    and `run_as_type`.
    """
    body = {
        "display_name": SPACE_TITLE,
        "description": SPACE_DESCRIPTION,
        "warehouse_id": DBX_WAREHOUSE_ID,
        "table_identifiers": [
            f"{DBX_CATALOG}.{CRM_SCHEMA}.{t}" for t in TABLES
        ],
        "run_as_type": "VIEWER",
    }
    r = client.post(f"{DBX_HOST}/api/2.0/data-rooms", json=body)
    if r.status_code == 200:
        data = r.json()
        return data.get("space_id") or data.get("id")
    log(f"create failed {r.status_code}: {r.text[:400]}", indent=1)
    return None


def add_sample_questions(client: httpx.Client, space_id: str) -> None:
    """
    Add each sample question as a SAMPLE_QUESTION curated-question on the
    space. Skips questions that already exist (by exact text match).
    """
    url = f"{DBX_HOST}/api/2.0/data-rooms/{space_id}/curated-questions"
    existing_texts: set[str] = set()
    r = client.get(url)
    if r.status_code == 200:
        for q in r.json().get("curated_questions", []):
            if q.get("question_type") == "SAMPLE_QUESTION":
                existing_texts.add(q.get("question_text", ""))

    added = 0
    for q in SAMPLE_QUESTIONS:
        if q in existing_texts:
            continue
        body = {
            "curated_question": {
                "question_text": q,
                "question_type": "SAMPLE_QUESTION",
                "conversation_type": "NORMAL",
            }
        }
        r = client.post(url, json=body)
        if r.status_code != 200:
            log(
                f"WARN: could not add sample question {q!r}: "
                f"{r.status_code} {r.text[:200]}",
                indent=1,
            )
            continue
        added += 1
    log(
        f"sample questions: {added} added, "
        f"{len(existing_texts)} already present",
        indent=1,
    )


def print_manual_instructions() -> None:
    print()
    print("=" * 70)
    print("MANUAL STEP REQUIRED (CRM Genie space)")
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
        print(f"       - {DBX_CATALOG}.{CRM_SCHEMA}.{t}")
    print("  7. Add these sample questions:")
    for q in SAMPLE_QUESTIONS:
        print(f"       - {q}")
    print("  8. Save and copy the space_id from the URL")
    print("     (looks like https://.../genie/rooms/<space_id>).")
    print()
    print("Then re-run this script with:")
    print(
        "   GENIE_SPACE_CRM_ID=<id> "
        "uv run python scripts/create_crm_genie_space.py"
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

        log("Step 1) Locating CRM Genie space...")
        space_id = os.environ.get("GENIE_SPACE_CRM_ID") or None
        if space_id:
            log(f"using GENIE_SPACE_CRM_ID from env: {space_id}", indent=1)
        else:
            existing = find_existing_space(client)
            if existing:
                space_id = existing
                log(
                    f"found existing space with matching title: {space_id}",
                    indent=1,
                )
            else:
                log("creating Genie space via data-rooms API...", indent=1)
                space_id = create_space(client)
                if space_id:
                    log(f"created space: {space_id}", indent=1)

        if not space_id:
            print_manual_instructions()
            return 2

        log("Step 2) Adding sample questions...")
        add_sample_questions(client, space_id)

        log("Step 3) Granting CAN_RUN to both SPs...")
        grant_can_run(client, space_id)

        print()
        print("=" * 70)
        print("CRM GENIE SPACE READY")
        print("=" * 70)
        print(f"GENIE_SPACE_CRM_ID={space_id}")
        print(f"UI: {DBX_HOST}/genie/rooms/{space_id}")
        print()
        print("Paste the following into your env / main-app config:")
        print()
        print(f"GENIE_SPACE_CRM_ID={space_id}")
        print(f'GENIE_SPACE_CRM_SCOPE="{SCOPE_DESCRIPTION}"')
        return 0
    except Exception as exc:
        print(f"\nFAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
