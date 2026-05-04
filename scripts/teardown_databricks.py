#!/usr/bin/env python
"""
Tears down everything setup_databricks.py / create_genie_space.py created.

Idempotent — silently skips things that no longer exist.

  - DROP SCHEMA abhilash_r.genie_demo CASCADE
  - DELETE the Genie space if GENIE_SPACE_ID is set in .env
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import httpx
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env"
load_dotenv(ENV_FILE)

DBX_HOST = os.environ["DBX_HOST"].rstrip("/")
DBX_PROFILE = os.environ["DBX_PROFILE"]
DBX_WAREHOUSE_ID = os.environ["DBX_WAREHOUSE_ID"]
DBX_CATALOG = os.environ["DBX_CATALOG"]
DBX_SCHEMA = os.environ["DBX_SCHEMA"]
GENIE_SPACE_ID = os.environ.get("GENIE_SPACE_ID", "").strip()

FQ_SCHEMA = f"{DBX_CATALOG}.{DBX_SCHEMA}"


def log(msg: str, indent: int = 0) -> None:
    print("  " * indent + msg, flush=True)


def run_sql(w: WorkspaceClient, statement: str, *, label: str) -> None:
    resp = w.statement_execution.execute_statement(
        statement=statement,
        warehouse_id=DBX_WAREHOUSE_ID,
        wait_timeout="30s",
    )
    state = resp.status.state if resp.status else None
    statement_id = resp.statement_id
    waited = 0
    while state in (StatementState.PENDING, StatementState.RUNNING):
        if waited > 90:
            raise RuntimeError(f"SQL timeout for {label}")
        time.sleep(2)
        waited += 2
        resp = w.statement_execution.get_statement(statement_id)
        state = resp.status.state if resp.status else None

    if state != StatementState.SUCCEEDED:
        err = resp.status.error if resp.status else None
        msg = f"{err.error_code}: {err.message}" if err else "unknown"
        raise RuntimeError(f"SQL failed for {label} ({state}): {msg}")


def drop_schema(w: WorkspaceClient) -> None:
    log(f"Dropping schema {FQ_SCHEMA} CASCADE (if exists)...")
    try:
        run_sql(
            w,
            f"DROP SCHEMA IF EXISTS {FQ_SCHEMA} CASCADE",
            label="drop schema",
        )
        log("schema dropped", indent=1)
    except Exception as exc:
        log(f"WARN: could not drop schema: {exc}", indent=1)


def delete_genie_space() -> None:
    if not GENIE_SPACE_ID:
        log("No GENIE_SPACE_ID in .env — skipping Genie space delete.")
        return
    log(f"Deleting Genie space {GENIE_SPACE_ID}...")
    try:
        import subprocess
        import json

        res = subprocess.run(
            ["databricks", "auth", "token", "--host", DBX_HOST],
            capture_output=True,
            text=True,
            check=True,
        )
        tok = json.loads(res.stdout)["access_token"]
        r = httpx.delete(
            f"{DBX_HOST}/api/2.0/genie/spaces/{GENIE_SPACE_ID}",
            headers={"Authorization": f"Bearer {tok}"},
            timeout=30.0,
        )
        if r.status_code in (200, 204):
            log("space deleted", indent=1)
        elif r.status_code == 404:
            log("space already gone", indent=1)
        else:
            log(
                f"WARN: unexpected status {r.status_code}: {r.text[:200]}",
                indent=1,
            )
    except Exception as exc:
        log(f"WARN: could not delete space: {exc}", indent=1)


def main() -> int:
    try:
        w = WorkspaceClient(profile=DBX_PROFILE)
        drop_schema(w)
        delete_genie_space()
        print()
        print("TEARDOWN COMPLETE")
        return 0
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
