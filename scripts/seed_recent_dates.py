#!/usr/bin/env python
"""Seed a small batch of NEW rows dated for the last two days into the demo tables.

The original `setup_databricks.py` and `setup_crm_databricks.py` scripts seed
historical data. This add-on script tops up the same tables with rows for the
last two dates so demos don't look stale.

Idempotent — every row this script inserts has an `R` (recent) prefix on its
ID. Re-running first DELETEs the `R*` rows, then re-inserts them.

Tables touched (RLS already in place, no grants needed):
  - abhilash_r.genie_demo.sales              (8 rows: 4/dealership × 2 dates)
  - abhilash_r.genie_demo.service_tickets    (6 rows: 3/dealership × 2 dates)
  - abhilash_r.genie_crm.leads               (8 rows)
  - abhilash_r.genie_crm.activities          (10 rows on existing opportunities)

Run:
  uv run python scripts/seed_recent_dates.py
"""

from __future__ import annotations

import os
import random
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from databricks.sdk import WorkspaceClient  # noqa: E402
from databricks.sdk.service.sql import StatementState  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

DBX_HOST = os.environ["DBX_HOST"]
DBX_WAREHOUSE_ID = os.environ["DBX_WAREHOUSE_ID"]
DBX_CATALOG = os.environ["DBX_CATALOG"]
SOURCE_SCHEMA = os.environ.get("DBX_SCHEMA", "genie_demo")
CRM_SCHEMA = os.environ.get("DBX_CRM_SCHEMA", "genie_crm")

FQ_SOURCE = f"{DBX_CATALOG}.{SOURCE_SCHEMA}"
FQ_CRM = f"{DBX_CATALOG}.{CRM_SCHEMA}"

random.seed(42)

# ---------------------------------------------------------------------------
# Reference data (kept in sync with setup_databricks.py / setup_crm_databricks.py)
# ---------------------------------------------------------------------------

DEALERS = ["NS001", "SR001"]

SALESPEOPLE_NS = ["Alice Manning", "Bob Peterson", "Cara Lin", "Dana Ruiz"]
SALESPEOPLE_SR = ["Eric Wong", "Faye Olsen", "Gus Patel", "Hina Brooks"]

FIRST_NAMES = ["Sage", "River", "Jordan", "Reese", "Morgan", "Taylor", "Jamie", "Quinn"]
LAST_NAMES = ["Patel", "Johnson", "Rossi", "Bennett", "Okafor", "Kowalski", "Hassan", "Singh"]

ISSUES = [
    "Brake pads replacement",
    "Oil change + filter",
    "AC compressor diagnosis",
    "Tire rotation + alignment",
    "Battery replacement",
    "Check-engine light follow-up",
]
TICKET_STATUSES = ["open", "in_progress", "resolved"]

LEAD_SOURCES = ["walk-in", "web", "referral", "ad"]
LEAD_STAGES = ["new", "contacted", "qualified"]

ACTIVITY_TYPES = ["call", "email", "demo", "test_drive", "quote"]
ACTIVITY_NOTES = [
    "Followed up on prior quote.",
    "Confirmed availability of trim.",
    "Customer asked about financing options.",
    "Set up test-drive appointment.",
    "Sent updated trade-in offer.",
]

# ---------------------------------------------------------------------------
# SQL helpers
# ---------------------------------------------------------------------------


def run_sql(w: WorkspaceClient, statement: str, *, label: str = "") -> None:
    exec = w.statement_execution.execute_statement(
        warehouse_id=DBX_WAREHOUSE_ID,
        statement=statement,
        wait_timeout="30s",
    )
    state = (exec.status.state if exec.status else None) or StatementState.SUCCEEDED
    if state != StatementState.SUCCEEDED:
        err = exec.status.error if exec.status else None
        raise RuntimeError(f"SQL '{label}' failed: {err}\n{statement[:300]}")
    if label:
        print(f"  {label} ✓")


def _sql_lit(v) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, str):
        return "'" + v.replace("'", "''") + "'"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, datetime):
        return f"TIMESTAMP '{v.strftime('%Y-%m-%d %H:%M:%S')}'"
    if isinstance(v, date):
        return f"DATE '{v.isoformat()}'"
    return "'" + str(v).replace("'", "''") + "'"


def insert_rows(
    w: WorkspaceClient, *, table: str, columns: list[str], rows: list[tuple], label: str
) -> None:
    if not rows:
        return
    col_list = ", ".join(columns)
    values = ",\n  ".join(
        "(" + ", ".join(_sql_lit(v) for v in row) + ")" for row in rows
    )
    run_sql(
        w,
        f"INSERT INTO {table} ({col_list}) VALUES\n  {values}",
        label=f"{label} ({len(rows)} rows)",
    )


def delete_recent(w: WorkspaceClient, *, table: str, id_col: str, prefix: str) -> None:
    run_sql(
        w,
        f"DELETE FROM {table} WHERE {id_col} LIKE '{prefix}%'",
        label=f"reset {table}.{id_col} LIKE '{prefix}%'",
    )


# ---------------------------------------------------------------------------
# Row generators (deterministic)
# ---------------------------------------------------------------------------


def gen_sales(dates: list[date], vehicles_by_dealer: dict[str, list[str]]) -> list[tuple]:
    rows: list[tuple] = []
    counter = 1
    for d in dates:
        for dealer in DEALERS:
            pool = vehicles_by_dealer.get(dealer) or []
            sps = SALESPEOPLE_NS if dealer == "NS001" else SALESPEOPLE_SR
            for _ in range(2):  # 2 sales per dealer per date
                vid = random.choice(pool) if pool else "V0001"
                price = round(random.uniform(15_000, 55_000), 2)
                customer = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
                salesperson = random.choice(sps)
                sale_id = f"R{d.strftime('%m%d')}S{counter:02d}"
                rows.append((sale_id, dealer, vid, d, price, salesperson, customer))
                counter += 1
    return rows


def gen_service_tickets(
    dates: list[date], vehicles_by_dealer: dict[str, list[str]]
) -> list[tuple]:
    rows: list[tuple] = []
    counter = 1
    for d in dates:
        for dealer in DEALERS:
            pool = vehicles_by_dealer.get(dealer) or []
            for _ in range(2):
                vid = random.choice(pool) if pool else "V0001"
                status = random.choice(TICKET_STATUSES)
                issue = random.choice(ISSUES)
                cost = round(random.uniform(60, 1200), 2)
                ticket_id = f"R{d.strftime('%m%d')}T{counter:02d}"
                rows.append((ticket_id, dealer, vid, d, status, issue, cost))
                counter += 1
    return rows


def gen_leads(dates: list[date]) -> list[tuple]:
    rows: list[tuple] = []
    counter = 1
    for d in dates:
        for dealer in DEALERS:
            sps = SALESPEOPLE_NS if dealer == "NS001" else SALESPEOPLE_SR
            for _ in range(2):
                first = random.choice(FIRST_NAMES)
                last = random.choice(LAST_NAMES)
                email = f"{first.lower()}.{last.lower()}@example.com"
                phone = f"555-{random.randint(1000, 9999)}"
                source = random.choice(LEAD_SOURCES)
                stage = random.choice(LEAD_STAGES)
                salesperson = random.choice(sps)
                created_at = datetime.combine(d, time(hour=random.randint(9, 17), minute=random.randint(0, 59)))
                last_activity_at = created_at + timedelta(hours=random.randint(0, 6))
                lead_id = f"R{d.strftime('%m%d')}L{counter:02d}"
                rows.append(
                    (
                        lead_id,
                        dealer,
                        first,
                        last,
                        email,
                        phone,
                        source,
                        stage,
                        salesperson,
                        created_at,
                        last_activity_at,
                    )
                )
                counter += 1
    return rows


def gen_activities(
    dates: list[date], opp_to_dealer: list[tuple[str, str]]
) -> list[tuple]:
    """Pin recent activities to existing opportunities (one per dealer per date)."""
    rows: list[tuple] = []
    counter = 1
    by_dealer: dict[str, list[str]] = {"NS001": [], "SR001": []}
    for opp_id, dealer in opp_to_dealer:
        by_dealer.setdefault(dealer, []).append(opp_id)
    for d in dates:
        for dealer in DEALERS:
            opps = by_dealer.get(dealer) or []
            if not opps:
                continue
            for _ in range(2):
                opp_id = random.choice(opps)
                act_type = random.choice(ACTIVITY_TYPES)
                ts = datetime.combine(d, time(hour=random.randint(9, 17), minute=random.randint(0, 59)))
                notes = random.choice(ACTIVITY_NOTES)
                outcome = random.choice(["positive", "neutral", "follow_up"])
                act_id = f"R{d.strftime('%m%d')}A{counter:02d}"
                rows.append((act_id, opp_id, dealer, act_type, ts, notes, outcome))
                counter += 1
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def fetch_existing(w: WorkspaceClient) -> tuple[dict[str, list[str]], list[tuple[str, str]]]:
    """Return (vehicle_id pool by dealer, list of (opportunity_id, dealership_id))."""
    vehicles_by_dealer: dict[str, list[str]] = {"NS001": [], "SR001": []}
    exec = w.statement_execution.execute_statement(
        warehouse_id=DBX_WAREHOUSE_ID,
        statement=f"SELECT vehicle_id, dealership_id FROM {FQ_SOURCE}.vehicles",
        wait_timeout="30s",
    )
    if exec.result and exec.result.data_array:
        for row in exec.result.data_array:
            dealer = row[1]
            vehicles_by_dealer.setdefault(dealer, []).append(row[0])

    opps: list[tuple[str, str]] = []
    exec = w.statement_execution.execute_statement(
        warehouse_id=DBX_WAREHOUSE_ID,
        statement=f"SELECT opportunity_id, dealership_id FROM {FQ_CRM}.opportunities",
        wait_timeout="30s",
    )
    if exec.result and exec.result.data_array:
        opps = [(r[0], r[1]) for r in exec.result.data_array]

    return vehicles_by_dealer, opps


def main() -> None:
    today = date.today()
    dates = [today - timedelta(days=1), today]
    print(f"Seeding recent rows for: {dates[0].isoformat()}, {dates[1].isoformat()}")
    print(f"Catalog: {DBX_CATALOG}  schemas: {SOURCE_SCHEMA}, {CRM_SCHEMA}")
    print()

    w = WorkspaceClient(host=DBX_HOST)

    print("Fetching existing reference rows…")
    vehicles_by_dealer, opps = fetch_existing(w)
    print(
        f"  vehicles: NS001={len(vehicles_by_dealer.get('NS001', []))} "
        f"SR001={len(vehicles_by_dealer.get('SR001', []))}; opportunities={len(opps)}"
    )

    sales_rows = gen_sales(dates, vehicles_by_dealer)
    ticket_rows = gen_service_tickets(dates, vehicles_by_dealer)
    lead_rows = gen_leads(dates)
    activity_rows = gen_activities(dates, opps)

    # Reset + insert each table.
    print("\nsales (genie_demo)")
    delete_recent(w, table=f"{FQ_SOURCE}.sales", id_col="sale_id", prefix="R")
    insert_rows(
        w,
        table=f"{FQ_SOURCE}.sales",
        columns=["sale_id", "dealership_id", "vehicle_id", "sale_date", "sale_price", "salesperson", "customer_name"],
        rows=sales_rows,
        label="insert recent sales",
    )

    print("\nservice_tickets (genie_demo)")
    delete_recent(w, table=f"{FQ_SOURCE}.service_tickets", id_col="ticket_id", prefix="R")
    insert_rows(
        w,
        table=f"{FQ_SOURCE}.service_tickets",
        columns=["ticket_id", "dealership_id", "vehicle_id", "opened_date", "status", "issue", "cost"],
        rows=ticket_rows,
        label="insert recent service tickets",
    )

    print("\nleads (genie_crm)")
    delete_recent(w, table=f"{FQ_CRM}.leads", id_col="lead_id", prefix="R")
    insert_rows(
        w,
        table=f"{FQ_CRM}.leads",
        columns=[
            "lead_id", "dealership_id", "first_name", "last_name", "email", "phone",
            "source", "stage", "assigned_salesperson", "created_at", "last_activity_at",
        ],
        rows=lead_rows,
        label="insert recent leads",
    )

    print("\nactivities (genie_crm)")
    delete_recent(w, table=f"{FQ_CRM}.activities", id_col="activity_id", prefix="R")
    insert_rows(
        w,
        table=f"{FQ_CRM}.activities",
        columns=["activity_id", "opportunity_id", "dealership_id", "activity_type", "timestamp", "notes", "outcome"],
        rows=activity_rows,
        label="insert recent activities",
    )

    print("\nDone.")
    print(
        f"  sales: +{len(sales_rows)}  service_tickets: +{len(ticket_rows)}  "
        f"leads: +{len(lead_rows)}  activities: +{len(activity_rows)}"
    )
    print("\nVerify (as the workspace owner):")
    print(
        f"  SELECT sale_date, COUNT(*) FROM {FQ_SOURCE}.sales "
        f"WHERE sale_date >= DATE'{dates[0].isoformat()}' GROUP BY sale_date ORDER BY 1;"
    )


if __name__ == "__main__":
    main()
