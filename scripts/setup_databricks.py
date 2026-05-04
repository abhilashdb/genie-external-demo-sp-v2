#!/usr/bin/env python
"""
Sets up the Databricks environment for the Genie Space SP demo.

Idempotent. Safe to re-run.

Steps:
  a) Verify connectivity + look up the two service principals by applicationId.
  b) CREATE SCHEMA IF NOT EXISTS abhilash_r.genie_demo.
  c) Seed synthetic tables: dealerships, vehicles, sales, service_tickets.
  d) Create UC row-filter function dealership_rls and apply to every table.
  e) Grant USE / SELECT / EXECUTE to both SPs.
  f) Verify RLS works by token-exchanging each SP and running COUNT(*).
  g) Print summary.

Run with:  .venv/bin/python scripts/setup_databricks.py
"""

from __future__ import annotations

import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import httpx
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState
from dotenv import load_dotenv


# --------------------------------------------------------------------------
# Env + constants
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

DBX_HOST = os.environ["DBX_HOST"].rstrip("/")
DBX_PROFILE = os.environ["DBX_PROFILE"]
DBX_WAREHOUSE_ID = os.environ["DBX_WAREHOUSE_ID"]
DBX_CATALOG = os.environ["DBX_CATALOG"]
DBX_SCHEMA = os.environ["DBX_SCHEMA"]

SP_NORTHSTAR_CLIENT_ID = os.environ["SP_NORTHSTAR_CLIENT_ID"]
SP_NORTHSTAR_SECRET = os.environ["SP_NORTHSTAR_SECRET"]
SP_SUNRISE_CLIENT_ID = os.environ["SP_SUNRISE_CLIENT_ID"]
SP_SUNRISE_SECRET = os.environ["SP_SUNRISE_SECRET"]

FQ_SCHEMA = f"{DBX_CATALOG}.{DBX_SCHEMA}"
POLICY_TABLE = f"{FQ_SCHEMA}.dealership_principals"

# Deterministic seed so repeated runs produce the same data.
random.seed(42)


def log(msg: str, indent: int = 0) -> None:
    prefix = "  " * indent
    print(f"{prefix}{msg}", flush=True)


# --------------------------------------------------------------------------
# SQL helpers
# --------------------------------------------------------------------------


def run_sql(
    w: WorkspaceClient, statement: str, *, label: str | None = None
) -> list[list]:
    """Execute a SQL statement synchronously; poll until terminal state."""
    resp = w.statement_execution.execute_statement(
        statement=statement,
        warehouse_id=DBX_WAREHOUSE_ID,
        wait_timeout="30s",
    )
    statement_id = resp.statement_id
    state = resp.status.state if resp.status else None

    # Poll if still running
    waited = 0
    while state in (StatementState.PENDING, StatementState.RUNNING):
        if waited > 120:
            raise RuntimeError(f"SQL timed out after 120s: {label or statement[:80]}")
        time.sleep(2)
        waited += 2
        resp = w.statement_execution.get_statement(statement_id)
        state = resp.status.state if resp.status else None

    if state != StatementState.SUCCEEDED:
        err = resp.status.error if resp.status else None
        detail = f"{err.error_code}: {err.message}" if err else "unknown error"
        raise RuntimeError(
            f"SQL failed ({state}) — {label or statement[:120]}\n  -> {detail}"
        )

    rows = []
    if resp.result and resp.result.data_array:
        rows = resp.result.data_array
    return rows


def run_sql_with_token(
    token: str, statement: str, *, label: str | None = None
) -> list[list]:
    """Run SQL via REST using an explicit bearer token (for SP verification)."""
    url = f"{DBX_HOST}/api/2.0/sql/statements"
    payload = {
        "statement": statement,
        "warehouse_id": DBX_WAREHOUSE_ID,
        "wait_timeout": "30s",
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=60.0) as client:
        r = client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        body = r.json()
        statement_id = body["statement_id"]
        state = body["status"]["state"]

        waited = 0
        while state in ("PENDING", "RUNNING"):
            if waited > 120:
                raise RuntimeError(f"SQL timed out: {label or statement[:80]}")
            time.sleep(2)
            waited += 2
            r = client.get(
                f"{DBX_HOST}/api/2.0/sql/statements/{statement_id}", headers=headers
            )
            r.raise_for_status()
            body = r.json()
            state = body["status"]["state"]

        if state != "SUCCEEDED":
            err = body["status"].get("error", {})
            raise RuntimeError(
                f"SQL failed ({state}) — {label or statement[:120]}\n  -> {err}"
            )
        return body.get("result", {}).get("data_array", []) or []


def oauth_token(client_id: str, client_secret: str) -> str:
    """Exchange SP client credentials for a workspace OAuth token."""
    url = f"{DBX_HOST}/oidc/v1/token"
    with httpx.Client(timeout=30.0) as client:
        r = client.post(
            url,
            data={
                "grant_type": "client_credentials",
                "scope": "all-apis",
            },
            auth=(client_id, client_secret),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        r.raise_for_status()
        return r.json()["access_token"]


# --------------------------------------------------------------------------
# Step (a) — Connectivity + SP lookup
# --------------------------------------------------------------------------


@dataclass
class SPInfo:
    label: str
    client_id: str
    display_name: str
    internal_id: str
    dealership_id: str


def lookup_sps(w: WorkspaceClient) -> tuple[SPInfo, SPInfo]:
    log("Step a) Verifying connectivity and looking up SPs...")
    me = w.current_user.me()
    log(f"connected as {me.user_name} @ {DBX_HOST}", indent=1)

    sps = {}
    for label, client_id, dealership_id in (
        ("northstar", SP_NORTHSTAR_CLIENT_ID, "NS001"),
        ("sunrise", SP_SUNRISE_CLIENT_ID, "SR001"),
    ):
        results = list(
            w.service_principals.list(filter=f"applicationId eq {client_id}")
        )
        if not results:
            raise RuntimeError(
                f"Service principal {client_id} ({label}) not found in workspace."
            )
        sp = results[0]
        info = SPInfo(
            label=label,
            client_id=client_id,
            display_name=sp.display_name or "",
            internal_id=str(sp.id),
            dealership_id=dealership_id,
        )
        sps[label] = info
        log(
            f"{label}: appId={info.client_id} id={info.internal_id} "
            f"displayName={info.display_name!r}",
            indent=1,
        )

    return sps["northstar"], sps["sunrise"]


# --------------------------------------------------------------------------
# Step (b) — Catalog / schema
# --------------------------------------------------------------------------


def ensure_schema(w: WorkspaceClient) -> None:
    log("Step b) Ensuring schema exists...")
    run_sql(
        w,
        f"CREATE SCHEMA IF NOT EXISTS {FQ_SCHEMA}",
        label="create schema",
    )
    log(f"schema {FQ_SCHEMA} ready", indent=1)


# --------------------------------------------------------------------------
# Step (c) — Synthetic data
# --------------------------------------------------------------------------


def _quote(val) -> str:
    if val is None:
        return "NULL"
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, date):
        return f"DATE '{val.isoformat()}'"
    s = str(val).replace("'", "''")
    return f"'{s}'"


def _insert_rows(
    w: WorkspaceClient, table: str, columns: list[str], rows: list[tuple]
) -> None:
    if not rows:
        return
    # Batch ~50 rows per INSERT to avoid overly long statements.
    col_list = ", ".join(columns)
    batch = 50
    for i in range(0, len(rows), batch):
        chunk = rows[i : i + batch]
        values = ",\n  ".join(
            "(" + ", ".join(_quote(v) for v in row) + ")" for row in chunk
        )
        run_sql(
            w,
            f"INSERT INTO {FQ_SCHEMA}.{table} ({col_list}) VALUES\n  {values}",
            label=f"insert into {table}",
        )


DEALERSHIPS = [
    ("NS001", "North Star Motors", "Midwest", "Chicago"),
    ("SR001", "Sunrise Auto Group", "West", "Los Angeles"),
]

MAKES_MODELS = [
    ("Toyota", "Camry"),
    ("Toyota", "RAV4"),
    ("Toyota", "Highlander"),
    ("Honda", "Accord"),
    ("Honda", "CR-V"),
    ("Honda", "Civic"),
    ("Ford", "F-150"),
    ("Ford", "Escape"),
    ("Ford", "Explorer"),
    ("Chevrolet", "Silverado"),
    ("Chevrolet", "Equinox"),
    ("Nissan", "Altima"),
    ("Nissan", "Rogue"),
    ("Hyundai", "Elantra"),
    ("Hyundai", "Tucson"),
    ("Tesla", "Model 3"),
    ("Tesla", "Model Y"),
    ("Subaru", "Outback"),
    ("Mazda", "CX-5"),
    ("Kia", "Sportage"),
]

SALESPEOPLE_NS = ["Alice Manning", "Bob Peterson", "Dana Ruiz", "Eric Wong"]
SALESPEOPLE_SR = ["Carol Jenkins", "Derek Ford", "Priya Shah", "Marco Rossi"]

FIRST_NAMES = [
    "Jordan",
    "Taylor",
    "Morgan",
    "Casey",
    "Jamie",
    "Avery",
    "Riley",
    "Quinn",
    "Sky",
    "Drew",
    "Parker",
    "Reese",
    "Emerson",
    "Sage",
    "Rowan",
]
LAST_NAMES = [
    "Chen",
    "Patel",
    "Garcia",
    "Nguyen",
    "Williams",
    "Johnson",
    "Rossi",
    "Kowalski",
    "Okafor",
    "Haines",
    "Bennett",
    "Torres",
]

ISSUES = [
    "Oil change",
    "Brake pad replacement",
    "Transmission leak",
    "Battery replacement",
    "Check engine light",
    "Tire rotation",
    "AC repair",
    "Coolant flush",
    "Suspension noise",
    "Windshield replacement",
]
STATUSES = ["OPEN", "IN_PROGRESS", "COMPLETED", "COMPLETED", "COMPLETED"]


def _gen_vehicles() -> list[tuple]:
    vehicles = []
    for i in range(20):
        make, model = MAKES_MODELS[i]
        # Split: first 10 -> NS001, next 10 -> SR001
        dealership = "NS001" if i < 10 else "SR001"
        year = random.randint(2019, 2025)
        price = round(random.uniform(18_000, 62_000), 2)
        stock = random.randint(1, 15)
        vehicles.append(
            (
                f"V{i + 1:03d}",
                dealership,
                make,
                model,
                year,
                price,
                stock,
            )
        )
    return vehicles


def _gen_sales(vehicles: list[tuple]) -> list[tuple]:
    sales = []
    today = date.today()
    ns_vehicles = [v for v in vehicles if v[1] == "NS001"]
    sr_vehicles = [v for v in vehicles if v[1] == "SR001"]
    for i in range(40):
        dealership = "NS001" if i % 2 == 0 else "SR001"
        pool = ns_vehicles if dealership == "NS001" else sr_vehicles
        vehicle = random.choice(pool)
        vid = vehicle[0]
        base_price = float(vehicle[5])
        # Sale price within +/- 8% of sticker
        sale_price = round(base_price * random.uniform(0.92, 1.03), 2)
        days_back = random.randint(1, 90)
        sale_date = today - timedelta(days=days_back)
        salesperson = random.choice(
            SALESPEOPLE_NS if dealership == "NS001" else SALESPEOPLE_SR
        )
        customer = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        sales.append(
            (
                f"S{i + 1:04d}",
                dealership,
                vid,
                sale_date,
                sale_price,
                salesperson,
                customer,
            )
        )
    return sales


def _gen_service_tickets(vehicles: list[tuple]) -> list[tuple]:
    tickets = []
    today = date.today()
    for i in range(15):
        dealership = "NS001" if i % 2 == 0 else "SR001"
        pool = [v for v in vehicles if v[1] == dealership]
        vehicle = random.choice(pool)
        vid = vehicle[0]
        days_back = random.randint(1, 60)
        opened = today - timedelta(days=days_back)
        status = random.choice(STATUSES)
        issue = random.choice(ISSUES)
        cost = round(random.uniform(60, 1800), 2)
        tickets.append(
            (
                f"T{i + 1:04d}",
                dealership,
                vid,
                opened,
                status,
                issue,
                cost,
            )
        )
    return tickets


def seed_tables(w: WorkspaceClient) -> None:
    log("Step c) Seeding synthetic data...")

    # --- dealerships ---
    run_sql(
        w,
        f"""
        CREATE OR REPLACE TABLE {FQ_SCHEMA}.dealerships (
          dealership_id STRING,
          name STRING,
          region STRING,
          city STRING
        )
        """.strip(),
        label="create dealerships",
    )
    _insert_rows(
        w,
        "dealerships",
        ["dealership_id", "name", "region", "city"],
        DEALERSHIPS,
    )
    log(f"dealerships: {len(DEALERSHIPS)} rows", indent=1)

    # --- vehicles ---
    run_sql(
        w,
        f"""
        CREATE OR REPLACE TABLE {FQ_SCHEMA}.vehicles (
          vehicle_id STRING,
          dealership_id STRING,
          make STRING,
          model STRING,
          year INT,
          price DECIMAL(10,2),
          stock INT
        )
        """.strip(),
        label="create vehicles",
    )
    vehicles = _gen_vehicles()
    _insert_rows(
        w,
        "vehicles",
        ["vehicle_id", "dealership_id", "make", "model", "year", "price", "stock"],
        vehicles,
    )
    log(f"vehicles: {len(vehicles)} rows", indent=1)

    # --- sales ---
    run_sql(
        w,
        f"""
        CREATE OR REPLACE TABLE {FQ_SCHEMA}.sales (
          sale_id STRING,
          dealership_id STRING,
          vehicle_id STRING,
          sale_date DATE,
          sale_price DECIMAL(10,2),
          salesperson STRING,
          customer_name STRING
        )
        """.strip(),
        label="create sales",
    )
    sales = _gen_sales(vehicles)
    _insert_rows(
        w,
        "sales",
        [
            "sale_id",
            "dealership_id",
            "vehicle_id",
            "sale_date",
            "sale_price",
            "salesperson",
            "customer_name",
        ],
        sales,
    )
    log(f"sales: {len(sales)} rows", indent=1)

    # --- service_tickets ---
    run_sql(
        w,
        f"""
        CREATE OR REPLACE TABLE {FQ_SCHEMA}.service_tickets (
          ticket_id STRING,
          dealership_id STRING,
          vehicle_id STRING,
          opened_date DATE,
          status STRING,
          issue STRING,
          cost DECIMAL(10,2)
        )
        """.strip(),
        label="create service_tickets",
    )
    tickets = _gen_service_tickets(vehicles)
    _insert_rows(
        w,
        "service_tickets",
        [
            "ticket_id",
            "dealership_id",
            "vehicle_id",
            "opened_date",
            "status",
            "issue",
            "cost",
        ],
        tickets,
    )
    log(f"service_tickets: {len(tickets)} rows", indent=1)


# --------------------------------------------------------------------------
# Step (d) — ABAC policy table + row-filter function + apply
# --------------------------------------------------------------------------


RLS_TABLES = ["dealerships", "vehicles", "sales", "service_tickets"]


def ensure_policy_table(w: WorkspaceClient, northstar: SPInfo, sunrise: SPInfo) -> None:
    """Create/refresh the dealership_principals mapping table.

    This is the ABAC attribute store: principal_id (SP applicationId as returned
    by current_user()) → dealership_id. The row filter reads it at query time
    instead of having SP identities hardcoded in SQL.
    """
    log("Step d.1) Ensuring dealership_principals policy table...")
    run_sql(
        w,
        f"""
        CREATE TABLE IF NOT EXISTS {POLICY_TABLE} (
          principal_id  STRING,
          dealership_id STRING
        )
        """.strip(),
        label="create policy table",
    )
    # Full replace on every run so the table stays in sync with .env.
    run_sql(w, f"DELETE FROM {POLICY_TABLE}", label="clear policy table")
    run_sql(
        w,
        f"""
        INSERT INTO {POLICY_TABLE} (principal_id, dealership_id) VALUES
          ('{northstar.client_id}', '{northstar.dealership_id}'),
          ('{sunrise.client_id}',   '{sunrise.dealership_id}')
        """.strip(),
        label="populate policy table",
    )
    log(f"policy table {POLICY_TABLE} ready (2 rows)", indent=1)


def apply_rls(w: WorkspaceClient) -> None:
    log("Step d.2) Creating ABAC row filter and applying to tables...")

    rls_sql = f"""
    CREATE OR REPLACE FUNCTION {FQ_SCHEMA}.dealership_rls(row_dealership_id STRING)
    RETURNS BOOLEAN
    RETURN EXISTS (
      SELECT 1
      FROM   {POLICY_TABLE}
      WHERE  principal_id  = current_user()
        AND  dealership_id = row_dealership_id
    ) OR is_account_group_member('tek_admins')
    """.strip()
    run_sql(w, rls_sql, label="create rls function")
    log("dealership_rls() created", indent=1)

    for tbl in RLS_TABLES:
        run_sql(
            w,
            f"ALTER TABLE {FQ_SCHEMA}.{tbl} "
            f"SET ROW FILTER {FQ_SCHEMA}.dealership_rls ON (dealership_id)",
            label=f"apply rls to {tbl}",
        )
        log(f"applied to {tbl}", indent=1)


# --------------------------------------------------------------------------
# Step (e) — Grants
# --------------------------------------------------------------------------


def apply_grants(w: WorkspaceClient, sps: Iterable[SPInfo]) -> None:
    log("Step e) Applying grants...")
    for sp in sps:
        principal = f"`{sp.client_id}`"
        for stmt in (
            f"GRANT USE CATALOG ON CATALOG {DBX_CATALOG} TO {principal}",
            f"GRANT USE SCHEMA ON SCHEMA {FQ_SCHEMA} TO {principal}",
            f"GRANT SELECT ON SCHEMA {FQ_SCHEMA} TO {principal}",
            f"GRANT EXECUTE ON FUNCTION {FQ_SCHEMA}.dealership_rls TO {principal}",
            # SPs must be able to read the policy table that the row filter queries.
            f"GRANT SELECT ON TABLE {POLICY_TABLE} TO {principal}",
        ):
            run_sql(w, stmt, label=f"grant ({sp.label})")
        log(f"grants applied for {sp.label} ({sp.client_id})", indent=1)


# --------------------------------------------------------------------------
# Step (f) — Verify RLS with SP tokens
# --------------------------------------------------------------------------


def verify_rls(
    w: WorkspaceClient, northstar: SPInfo, sunrise: SPInfo
) -> tuple[int, int, int]:
    log("Step f) Verifying RLS with SP tokens...")

    total = int(run_sql(w, f"SELECT COUNT(*) FROM {FQ_SCHEMA}.sales")[0][0])
    log(f"total sales rows (catalog owner): {total}", indent=1)

    # Show what the policy table contains (as admin).
    policy_rows = run_sql(w, f"SELECT principal_id, dealership_id FROM {POLICY_TABLE}", label="read policy table")
    log(f"policy table rows: {policy_rows}", indent=1)

    counts = {}
    for sp, secret in (
        (northstar, SP_NORTHSTAR_SECRET),
        (sunrise, SP_SUNRISE_SECRET),
    ):
        try:
            tok = oauth_token(sp.client_id, secret)

            # 1. What does current_user() return for this SP token?
            cu = run_sql_with_token(tok, "SELECT current_user()", label=f"{sp.label} current_user")
            log(f"{sp.label} current_user() = {cu}", indent=1)

            # 2. Does the policy table return a row for this SP's identity?
            lookup = run_sql_with_token(
                tok,
                f"SELECT principal_id, dealership_id FROM {POLICY_TABLE} WHERE principal_id = current_user()",
                label=f"{sp.label} policy lookup",
            )
            log(f"{sp.label} policy lookup = {lookup}", indent=1)

            # 3. Actual filtered count.
            rows = run_sql_with_token(
                tok,
                f"SELECT COUNT(*) FROM {FQ_SCHEMA}.sales",
                label=f"SP {sp.label} count",
            )
            n = int(rows[0][0])
        except Exception as exc:
            log(f"WARN: could not query as {sp.label}: {exc}", indent=1)
            n = -1
        counts[sp.label] = n
        log(f"{sp.label} SP sees {n} sales rows", indent=1)

    ns, sr = counts.get("northstar", -1), counts.get("sunrise", -1)
    if ns >= 0 and sr >= 0 and ns == sr and ns == total:
        log(
            "WARNING: both SPs see all rows — RLS did not take effect!",
            indent=1,
        )
    return total, ns, sr


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    try:
        w = WorkspaceClient(profile=DBX_PROFILE)
        northstar, sunrise = lookup_sps(w)
        ensure_schema(w)
        seed_tables(w)
        ensure_policy_table(w, northstar, sunrise)
        apply_rls(w)
        apply_grants(w, [northstar, sunrise])
        total, ns_count, sr_count = verify_rls(w, northstar, sunrise)

        print()
        print("=" * 60)
        print("SETUP COMPLETE")
        print("=" * 60)
        print(f"Schema: {FQ_SCHEMA}")
        print(f"Tables: {', '.join(RLS_TABLES)}")
        print("RLS:    applied via dealership_rls()")
        print(
            f"Grants: {northstar.client_id} (northstar), "
            f"{sunrise.client_id} (sunrise)"
        )
        print(
            f"Rows:   total sales={total} | "
            f"northstar SP sees {ns_count} | sunrise SP sees {sr_count}"
        )
        print()
        print("SPs for Agent B:")
        print(
            f"  northstar -> appId={northstar.client_id} "
            f"internal_id={northstar.internal_id} "
            f"displayName={northstar.display_name!r}"
        )
        print(
            f"  sunrise   -> appId={sunrise.client_id} "
            f"internal_id={sunrise.internal_id} "
            f"displayName={sunrise.display_name!r}"
        )
        return 0
    except Exception as exc:
        print(f"\nFAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
