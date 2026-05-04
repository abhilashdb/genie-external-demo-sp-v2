#!/usr/bin/env python
"""
Sets up a SECOND Databricks schema (abhilash_r.genie_crm) for the CRM Genie
space demo. Mirrors setup_databricks.py in style + RLS pattern.

Idempotent. Safe to re-run.

Steps:
  a) Verify connectivity + look up the two service principals by applicationId.
  b) Pull the real dealerships + salespeople from abhilash_r.genie_demo so
     the synthetic CRM data reconciles with the sales/service Genie space.
  c) CREATE SCHEMA IF NOT EXISTS abhilash_r.genie_crm.
  d) Seed synthetic CRM tables:
       leads, opportunities, activities, sales_targets, forecasts.
  e) Create row-filter function abhilash_r.genie_crm.dealership_rls and
     apply it to every table on (dealership_id).
  f) GRANT USE CATALOG / USE SCHEMA / SELECT / EXECUTE to both SPs.
  g) Verify RLS works by token-exchanging each SP and running COUNT(*)
     against leads.
  h) Print summary.

Run with:  uv run python scripts/setup_crm_databricks.py
           (or .venv/bin/python scripts/setup_crm_databricks.py)
"""

from __future__ import annotations

import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
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
# NOTE: we reuse DBX_CATALOG but target a *new* schema for CRM tables.
SOURCE_SCHEMA = os.environ.get("DBX_SCHEMA", "genie_demo")
CRM_SCHEMA = "genie_crm"

SP_NORTHSTAR_CLIENT_ID = os.environ["SP_NORTHSTAR_CLIENT_ID"]
SP_NORTHSTAR_SECRET = os.environ["SP_NORTHSTAR_SECRET"]
SP_SUNRISE_CLIENT_ID = os.environ["SP_SUNRISE_CLIENT_ID"]
SP_SUNRISE_SECRET = os.environ["SP_SUNRISE_SECRET"]

FQ_SOURCE = f"{DBX_CATALOG}.{SOURCE_SCHEMA}"
FQ_CRM = f"{DBX_CATALOG}.{CRM_SCHEMA}"
POLICY_TABLE = f"{FQ_CRM}.dealership_principals"

# Deterministic seed so repeated runs produce the same data.
random.seed(1337)


def log(msg: str, indent: int = 0) -> None:
    prefix = "  " * indent
    print(f"{prefix}{msg}", flush=True)


# --------------------------------------------------------------------------
# SQL helpers (same shape as setup_databricks.py)
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

    waited = 0
    while state in (StatementState.PENDING, StatementState.RUNNING):
        if waited > 180:
            raise RuntimeError(
                f"SQL timed out after 180s: {label or statement[:80]}"
            )
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
                f"{DBX_HOST}/api/2.0/sql/statements/{statement_id}",
                headers=headers,
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
# Step (a) — SP lookup
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
# Step (b) — Pull real reference data from genie_demo
# --------------------------------------------------------------------------


@dataclass
class Reference:
    dealerships: list[tuple[str, str, str, str]]  # (id, name, region, city)
    salespeople_by_dealership: dict[str, list[str]]
    sales_by_dealer_month_sp: dict[tuple[str, str, str], tuple[int, float]]
    # key: (dealership_id, yyyy-mm, salesperson) -> (units, revenue)


def fetch_reference_data(w: WorkspaceClient) -> Reference:
    log("Step b) Pulling reference data from genie_demo...")

    dealerships_rows = run_sql(
        w,
        f"SELECT dealership_id, name, region, city FROM {FQ_SOURCE}.dealerships "
        "ORDER BY dealership_id",
        label="fetch dealerships",
    )
    dealerships = [
        (r[0], r[1], r[2], r[3]) for r in dealerships_rows
    ]
    log(f"dealerships: {len(dealerships)}", indent=1)

    sp_rows = run_sql(
        w,
        f"SELECT DISTINCT dealership_id, salesperson FROM {FQ_SOURCE}.sales "
        "ORDER BY dealership_id, salesperson",
        label="fetch salespeople",
    )
    salespeople: dict[str, list[str]] = {}
    for dealer_id, sp_name in sp_rows:
        salespeople.setdefault(dealer_id, []).append(sp_name)
    for d, lst in salespeople.items():
        log(f"{d}: {len(lst)} salespeople — {', '.join(lst)}", indent=1)

    sales_rows = run_sql(
        w,
        f"""
        SELECT dealership_id,
               date_format(sale_date, 'yyyy-MM') AS ym,
               salesperson,
               COUNT(*)                           AS units,
               SUM(sale_price)                    AS revenue
        FROM {FQ_SOURCE}.sales
        GROUP BY dealership_id, ym, salesperson
        """.strip(),
        label="fetch sales agg",
    )
    sales_map: dict[tuple[str, str, str], tuple[int, float]] = {}
    for d_id, ym, sp_name, units, revenue in sales_rows:
        sales_map[(d_id, ym, sp_name)] = (int(units), float(revenue))
    log(f"historical (dealer,month,sp) agg rows: {len(sales_map)}", indent=1)

    return Reference(
        dealerships=dealerships,
        salespeople_by_dealership=salespeople,
        sales_by_dealer_month_sp=sales_map,
    )


# --------------------------------------------------------------------------
# Step (c) — Schema
# --------------------------------------------------------------------------


def ensure_schema(w: WorkspaceClient) -> None:
    log(f"Step c) Ensuring schema {FQ_CRM} exists...")
    run_sql(
        w,
        f"CREATE SCHEMA IF NOT EXISTS {FQ_CRM}",
        label="create schema",
    )
    log(f"schema {FQ_CRM} ready", indent=1)


# --------------------------------------------------------------------------
# Step (d) — Synthetic CRM data
# --------------------------------------------------------------------------


def _quote(val) -> str:
    if val is None:
        return "NULL"
    if isinstance(val, bool):
        return "TRUE" if val else "FALSE"
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, datetime):
        return f"TIMESTAMP '{val.strftime('%Y-%m-%d %H:%M:%S')}'"
    if isinstance(val, date):
        return f"DATE '{val.isoformat()}'"
    s = str(val).replace("'", "''")
    return f"'{s}'"


def _insert_rows(
    w: WorkspaceClient,
    table: str,
    columns: list[str],
    rows: list[tuple],
    batch: int = 40,
) -> None:
    if not rows:
        return
    col_list = ", ".join(columns)
    for i in range(0, len(rows), batch):
        chunk = rows[i : i + batch]
        values = ",\n  ".join(
            "(" + ", ".join(_quote(v) for v in row) + ")" for row in chunk
        )
        run_sql(
            w,
            f"INSERT INTO {FQ_CRM}.{table} ({col_list}) VALUES\n  {values}",
            label=f"insert into {table}",
        )


# Static pools used for lead / activity text (distinct from the sales demo
# so the CRM space feels like a different dataset).
FIRST_NAMES = [
    "Ava", "Noah", "Mia", "Liam", "Zoe", "Ethan", "Isla", "Lucas",
    "Aria", "Kai", "Nora", "Owen", "Luna", "Ezra", "Maya", "Finn",
    "Ruby", "Arlo", "Cleo", "Jude", "Hazel", "Theo", "Iris", "Milo",
]
LAST_NAMES = [
    "Alvarez", "Brooks", "Cho", "Diaz", "Evans", "Fischer", "Gupta",
    "Haider", "Ibarra", "Joshi", "Kaur", "Lopez", "Mehta", "Nakamura",
    "Ortiz", "Park", "Quinn", "Rao", "Saito", "Tran", "Udo", "Vega",
]
DOMAINS = ["example.com", "mail.test", "demo.io", "acme.co", "nomail.dev"]
SOURCES = ["walk-in", "web", "referral", "ad", "event"]
LEAD_STAGES = ["new", "contacted", "qualified", "disqualified"]
# Weight stages so most leads are still early in the funnel.
LEAD_STAGE_WEIGHTS = [0.35, 0.30, 0.25, 0.10]
OPP_STAGES = ["prospecting", "demo", "negotiation", "won", "lost"]
OPP_STAGE_WEIGHTS = [0.28, 0.22, 0.18, 0.18, 0.14]
ACTIVITY_TYPES = ["call", "email", "demo", "test_drive", "quote"]
OUTCOMES = [
    "positive",
    "neutral",
    "no_response",
    "requested_quote",
    "scheduled_followup",
    "closed",
]
MAKES_MODELS = [
    ("Toyota", "Camry"),
    ("Toyota", "RAV4"),
    ("Toyota", "Highlander"),
    ("Honda", "Accord"),
    ("Honda", "CR-V"),
    ("Ford", "F-150"),
    ("Ford", "Escape"),
    ("Chevrolet", "Silverado"),
    ("Nissan", "Rogue"),
    ("Tesla", "Model Y"),
    ("Tesla", "Model 3"),
    ("Subaru", "Outback"),
    ("Mazda", "CX-5"),
    ("Kia", "Sportage"),
]
NOTE_SNIPPETS = [
    "Customer browsing inventory, interested in SUVs.",
    "Left voicemail, awaiting callback.",
    "Sent updated trade-in offer.",
    "Wants financing options under 4.9% APR.",
    "Shopping multiple dealerships — aggressive on price.",
    "Ready to close pending spouse approval.",
    "Test drive went well, considering options.",
    "Prefers hybrid or EV.",
    "Follow up next week after paycheck.",
    "Interested in extended warranty.",
    "Needs trade appraisal.",
    "Ghosted after quote sent.",
]


def _now() -> datetime:
    return datetime.now()


def _gen_leads(
    ref: Reference, per_dealership: int = 130
) -> list[tuple]:
    """Generate leads referencing real dealership_ids + salespeople."""
    leads: list[tuple] = []
    today = date.today()
    lead_idx = 0
    for dealer in ref.dealerships:
        dealership_id = dealer[0]
        sps = ref.salespeople_by_dealership.get(dealership_id, [])
        if not sps:
            continue
        for _ in range(per_dealership):
            lead_idx += 1
            fn = random.choice(FIRST_NAMES)
            ln = random.choice(LAST_NAMES)
            email = f"{fn.lower()}.{ln.lower()}{lead_idx}@{random.choice(DOMAINS)}"
            phone = (
                f"+1-{random.randint(200, 999)}-"
                f"{random.randint(200, 999)}-{random.randint(1000, 9999)}"
            )
            source = random.choice(SOURCES)
            stage = random.choices(LEAD_STAGES, weights=LEAD_STAGE_WEIGHTS)[0]
            salesperson = random.choice(sps)
            days_back = random.randint(0, 120)
            created_at = datetime.combine(
                today - timedelta(days=days_back),
                datetime.min.time(),
            ) + timedelta(minutes=random.randint(0, 24 * 60 - 1))
            # Last activity: somewhere between created_at and now, but leave
            # a ~20% population with a stale gap of 30+ days so the sample
            # question "which leads haven't had activity in 30+ days?" has
            # results for both dealerships.
            if random.random() < 0.22:
                last_activity_at = created_at + timedelta(
                    days=random.randint(0, 5)
                )
            else:
                max_recent = min(days_back, 29)
                last_activity_at = datetime.combine(
                    today - timedelta(days=random.randint(0, max_recent)),
                    datetime.min.time(),
                ) + timedelta(minutes=random.randint(0, 24 * 60 - 1))
                if last_activity_at < created_at:
                    last_activity_at = created_at
            leads.append(
                (
                    f"L{lead_idx:05d}",
                    dealership_id,
                    fn,
                    ln,
                    email,
                    phone,
                    source,
                    stage,
                    salesperson,
                    created_at,
                    last_activity_at,
                )
            )
    return leads


def _gen_opportunities(
    ref: Reference, leads: list[tuple]
) -> list[tuple]:
    """One opportunity for ~60% of qualified/contacted leads."""
    opps: list[tuple] = []
    today = date.today()
    opp_idx = 0
    for lead in leads:
        (
            lead_id,
            dealership_id,
            _fn,
            _ln,
            _email,
            _phone,
            _source,
            stage,
            salesperson,
            created_at,
            _last_activity_at,
        ) = lead
        if stage == "disqualified":
            continue
        if stage == "new" and random.random() > 0.25:
            continue
        if stage == "contacted" and random.random() > 0.55:
            continue
        if stage == "qualified" and random.random() > 0.85:
            continue
        opp_idx += 1
        opp_stage = random.choices(OPP_STAGES, weights=OPP_STAGE_WEIGHTS)[0]
        make, model = random.choice(MAKES_MODELS)
        expected_value = round(random.uniform(18_000, 68_000), 2)
        opp_created_at = created_at + timedelta(
            days=random.randint(0, 7),
            minutes=random.randint(0, 24 * 60 - 1),
        )
        # expected_close_date: near-future biased, but some historical
        # records for won/lost.
        if opp_stage in ("won", "lost"):
            close_days = random.randint(-60, 10)
        else:
            close_days = random.randint(3, 75)
        expected_close_date = today + timedelta(days=close_days)
        closed_at = None
        if opp_stage in ("won", "lost"):
            closed_at = opp_created_at + timedelta(
                days=random.randint(3, 45),
                minutes=random.randint(0, 24 * 60 - 1),
            )
        opps.append(
            (
                f"O{opp_idx:05d}",
                lead_id,
                dealership_id,
                salesperson,
                opp_stage,
                expected_close_date,
                expected_value,
                make,
                model,
                opp_created_at,
                closed_at,
            )
        )
    return opps


def _gen_activities(opps: list[tuple]) -> list[tuple]:
    """1-5 activities per opportunity."""
    activities: list[tuple] = []
    idx = 0
    for opp in opps:
        (
            opp_id,
            _lead_id,
            _dealership_id,
            _salesperson,
            _stage,
            _exp_close,
            _exp_value,
            _make,
            _model,
            opp_created_at,
            closed_at,
        ) = opp
        n_acts = random.randint(1, 5)
        end = closed_at or datetime.now()
        total_span = max((end - opp_created_at).total_seconds(), 60 * 60)
        for _ in range(n_acts):
            idx += 1
            offset = random.uniform(0, total_span)
            ts = opp_created_at + timedelta(seconds=offset)
            act_type = random.choice(ACTIVITY_TYPES)
            notes = random.choice(NOTE_SNIPPETS)
            outcome = random.choice(OUTCOMES)
            activities.append(
                (
                    f"A{idx:06d}",
                    opp_id,
                    act_type,
                    ts,
                    notes,
                    outcome,
                )
            )
    return activities


def _iter_months(start: date, count: int) -> list[date]:
    """Yield first-of-month dates, starting from start's month, for count months."""
    months = []
    y, m = start.year, start.month
    for _ in range(count):
        months.append(date(y, m, 1))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def _gen_sales_targets(ref: Reference) -> list[tuple]:
    """
    For each (dealership, salesperson), emit a target per month across the
    historical + current window. Historical months pull "achieved_*" from
    the real sales table so the two Genie spaces reconcile.
    """
    targets: list[tuple] = []
    idx = 0
    today = date.today()
    # Historical window comes from actual sales months, plus 2 future months
    # with achieved=0 for "this month / next month" targets.
    history_months = sorted({k[1] for k in ref.sales_by_dealer_month_sp.keys()})
    # Convert to date objects
    history_dates = [
        date(int(ym.split("-")[0]), int(ym.split("-")[1]), 1)
        for ym in history_months
    ]
    # Append two future months (for forecast alignment)
    if history_dates:
        last = history_dates[-1]
        y, m = last.year, last.month
        for _ in range(2):
            m += 1
            if m > 12:
                m = 1
                y += 1
            history_dates.append(date(y, m, 1))

    for dealer in ref.dealerships:
        dealership_id = dealer[0]
        sps = ref.salespeople_by_dealership.get(dealership_id, [])
        for sp_name in sps:
            for month in history_dates:
                idx += 1
                ym = month.strftime("%Y-%m")
                # Target: around 1.1x avg historical revenue (with noise),
                # target units around ceil(avg historical units * 1.1).
                target_revenue = round(random.uniform(120_000, 280_000), 2)
                target_units = random.randint(4, 12)
                units, revenue = ref.sales_by_dealer_month_sp.get(
                    (dealership_id, ym, sp_name), (0, 0.0)
                )
                # For future months, zero achieved so "above target this month"
                # queries reflect the real partial month if current.
                if month > today.replace(day=1):
                    units, revenue = 0, 0.0
                targets.append(
                    (
                        f"TGT{idx:05d}",
                        sp_name,
                        dealership_id,
                        month,
                        target_revenue,
                        target_units,
                        round(float(revenue), 2),
                        int(units),
                    )
                )
    return targets


def _gen_forecasts(ref: Reference) -> list[tuple]:
    """Per-dealership monthly forecasts: past 3 + next 3 months, one row per month."""
    forecasts: list[tuple] = []
    idx = 0
    today = date.today()
    # Window: 3 months back + current + 3 months fwd = 7 months.
    start = date(today.year, today.month, 1)
    # Subtract 3 months
    y, m = start.year, start.month
    for _ in range(3):
        m -= 1
        if m < 1:
            m = 12
            y -= 1
    start = date(y, m, 1)
    months = _iter_months(start, 7)
    as_of = today
    for dealer in ref.dealerships:
        dealership_id = dealer[0]
        # Base each forecast roughly on historical reality + growth.
        historical_units = [
            u for (d_id, _ym, _sp), (u, _r) in ref.sales_by_dealer_month_sp.items()
            if d_id == dealership_id
        ]
        avg_units = max(sum(historical_units) / max(len(historical_units), 1), 5)
        historical_rev = [
            r for (d_id, _ym, _sp), (_u, r) in ref.sales_by_dealer_month_sp.items()
            if d_id == dealership_id
        ]
        avg_rev = max(
            sum(historical_rev) / max(len(historical_rev), 1), 50_000.0
        )
        for month in months:
            idx += 1
            # Units forecast scaled 8-14x the per-sp avg (rough monthly total).
            fc_units = int(avg_units * random.uniform(6, 10))
            fc_rev = round(avg_rev * random.uniform(6, 10), 2)
            # Confidence: recent months higher, distant months lower.
            delta_months = (month.year - today.year) * 12 + (
                month.month - today.month
            )
            if delta_months <= 0:
                confidence = round(random.uniform(88, 98), 1)
            elif delta_months == 1:
                confidence = round(random.uniform(72, 85), 1)
            elif delta_months == 2:
                confidence = round(random.uniform(58, 72), 1)
            else:
                confidence = round(random.uniform(45, 60), 1)
            forecasts.append(
                (
                    f"F{idx:05d}",
                    dealership_id,
                    month,
                    fc_rev,
                    fc_units,
                    confidence,
                    as_of,
                )
            )
    return forecasts


def seed_tables(w: WorkspaceClient, ref: Reference) -> dict[str, int]:
    log("Step d) Seeding synthetic CRM data...")

    counts: dict[str, int] = {}

    # --- leads ---
    run_sql(
        w,
        f"""
        CREATE OR REPLACE TABLE {FQ_CRM}.leads (
          lead_id STRING,
          dealership_id STRING,
          first_name STRING,
          last_name STRING,
          email STRING,
          phone STRING,
          source STRING,
          stage STRING,
          assigned_salesperson STRING,
          created_at TIMESTAMP,
          last_activity_at TIMESTAMP
        )
        """.strip(),
        label="create leads",
    )
    leads = _gen_leads(ref, per_dealership=130)
    _insert_rows(
        w,
        "leads",
        [
            "lead_id",
            "dealership_id",
            "first_name",
            "last_name",
            "email",
            "phone",
            "source",
            "stage",
            "assigned_salesperson",
            "created_at",
            "last_activity_at",
        ],
        leads,
    )
    counts["leads"] = len(leads)
    log(f"leads: {len(leads)} rows", indent=1)

    # --- opportunities ---
    run_sql(
        w,
        f"""
        CREATE OR REPLACE TABLE {FQ_CRM}.opportunities (
          opportunity_id STRING,
          lead_id STRING,
          dealership_id STRING,
          salesperson STRING,
          stage STRING,
          expected_close_date DATE,
          expected_value_usd DECIMAL(12,2),
          vehicle_make STRING,
          vehicle_model STRING,
          created_at TIMESTAMP,
          closed_at TIMESTAMP
        )
        """.strip(),
        label="create opportunities",
    )
    opps = _gen_opportunities(ref, leads)
    _insert_rows(
        w,
        "opportunities",
        [
            "opportunity_id",
            "lead_id",
            "dealership_id",
            "salesperson",
            "stage",
            "expected_close_date",
            "expected_value_usd",
            "vehicle_make",
            "vehicle_model",
            "created_at",
            "closed_at",
        ],
        opps,
    )
    counts["opportunities"] = len(opps)
    log(f"opportunities: {len(opps)} rows", indent=1)

    # --- activities ---
    # NOTE: activities doesn't have dealership_id in the task spec. But we
    # need it for RLS. Add it (copied from the parent opportunity) so the
    # row filter can be applied uniformly.
    run_sql(
        w,
        f"""
        CREATE OR REPLACE TABLE {FQ_CRM}.activities (
          activity_id STRING,
          opportunity_id STRING,
          dealership_id STRING,
          activity_type STRING,
          timestamp TIMESTAMP,
          notes STRING,
          outcome STRING
        )
        """.strip(),
        label="create activities",
    )
    acts = _gen_activities(opps)
    # Enrich each activity row with its parent's dealership_id.
    opp_to_dealer = {opp[0]: opp[2] for opp in opps}
    activities_full: list[tuple] = []
    for a in acts:
        (a_id, opp_id, act_type, ts, notes, outcome) = a
        activities_full.append(
            (
                a_id,
                opp_id,
                opp_to_dealer.get(opp_id, "UNKNOWN"),
                act_type,
                ts,
                notes,
                outcome,
            )
        )
    _insert_rows(
        w,
        "activities",
        [
            "activity_id",
            "opportunity_id",
            "dealership_id",
            "activity_type",
            "timestamp",
            "notes",
            "outcome",
        ],
        activities_full,
    )
    counts["activities"] = len(activities_full)
    log(f"activities: {len(activities_full)} rows", indent=1)

    # --- sales_targets ---
    run_sql(
        w,
        f"""
        CREATE OR REPLACE TABLE {FQ_CRM}.sales_targets (
          target_id STRING,
          salesperson STRING,
          dealership_id STRING,
          month DATE,
          target_revenue_usd DECIMAL(12,2),
          target_units INT,
          achieved_revenue_usd DECIMAL(12,2),
          achieved_units INT
        )
        """.strip(),
        label="create sales_targets",
    )
    targets = _gen_sales_targets(ref)
    _insert_rows(
        w,
        "sales_targets",
        [
            "target_id",
            "salesperson",
            "dealership_id",
            "month",
            "target_revenue_usd",
            "target_units",
            "achieved_revenue_usd",
            "achieved_units",
        ],
        targets,
    )
    counts["sales_targets"] = len(targets)
    log(f"sales_targets: {len(targets)} rows", indent=1)

    # --- forecasts ---
    run_sql(
        w,
        f"""
        CREATE OR REPLACE TABLE {FQ_CRM}.forecasts (
          forecast_id STRING,
          dealership_id STRING,
          month DATE,
          forecast_revenue_usd DECIMAL(12,2),
          forecast_units INT,
          confidence_pct DECIMAL(5,2),
          as_of_date DATE
        )
        """.strip(),
        label="create forecasts",
    )
    forecasts = _gen_forecasts(ref)
    _insert_rows(
        w,
        "forecasts",
        [
            "forecast_id",
            "dealership_id",
            "month",
            "forecast_revenue_usd",
            "forecast_units",
            "confidence_pct",
            "as_of_date",
        ],
        forecasts,
    )
    counts["forecasts"] = len(forecasts)
    log(f"forecasts: {len(forecasts)} rows", indent=1)

    return counts


# --------------------------------------------------------------------------
# Step (e) — ABAC policy table + row-filter function + apply
# --------------------------------------------------------------------------


RLS_TABLES = ["leads", "opportunities", "activities", "sales_targets", "forecasts"]


def ensure_policy_table(w: WorkspaceClient, northstar: SPInfo, sunrise: SPInfo) -> None:
    """Create/refresh the dealership_principals mapping table for the CRM schema."""
    log("Step e.1) Ensuring dealership_principals policy table...")
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
    log("Step e.2) Creating ABAC row filter and applying to tables...")

    rls_sql = f"""
    CREATE OR REPLACE FUNCTION {FQ_CRM}.dealership_rls(row_dealership_id STRING)
    RETURNS BOOLEAN
    RETURN EXISTS (
      SELECT 1
      FROM   {POLICY_TABLE}
      WHERE  principal_id  = current_user()
        AND  dealership_id = row_dealership_id
    ) OR is_account_group_member('tek_admins')
    """.strip()
    run_sql(w, rls_sql, label="create rls function")
    log(f"{FQ_CRM}.dealership_rls() created", indent=1)

    for tbl in RLS_TABLES:
        run_sql(
            w,
            f"ALTER TABLE {FQ_CRM}.{tbl} "
            f"SET ROW FILTER {FQ_CRM}.dealership_rls ON (dealership_id)",
            label=f"apply rls to {tbl}",
        )
        log(f"applied to {tbl}", indent=1)


# --------------------------------------------------------------------------
# Step (f) — Grants
# --------------------------------------------------------------------------


def apply_grants(w: WorkspaceClient, sps: Iterable[SPInfo]) -> None:
    log("Step f) Applying grants...")
    for sp in sps:
        principal = f"`{sp.client_id}`"
        for stmt in (
            f"GRANT USE CATALOG ON CATALOG {DBX_CATALOG} TO {principal}",
            f"GRANT USE SCHEMA ON SCHEMA {FQ_CRM} TO {principal}",
            f"GRANT SELECT ON SCHEMA {FQ_CRM} TO {principal}",
            f"GRANT EXECUTE ON FUNCTION {FQ_CRM}.dealership_rls TO {principal}",
            f"GRANT SELECT ON TABLE {POLICY_TABLE} TO {principal}",
        ):
            run_sql(w, stmt, label=f"grant ({sp.label})")
        log(f"grants applied for {sp.label} ({sp.client_id})", indent=1)


# --------------------------------------------------------------------------
# Step (g) — Verify RLS with SP tokens
# --------------------------------------------------------------------------


def verify_rls(
    w: WorkspaceClient, northstar: SPInfo, sunrise: SPInfo
) -> tuple[int, int, int]:
    log("Step g) Verifying RLS with SP tokens...")

    total = int(run_sql(w, f"SELECT COUNT(*) FROM {FQ_CRM}.leads")[0][0])
    log(f"total leads rows (catalog owner): {total}", indent=1)

    counts: dict[str, int] = {}
    for sp, secret in (
        (northstar, SP_NORTHSTAR_SECRET),
        (sunrise, SP_SUNRISE_SECRET),
    ):
        try:
            tok = oauth_token(sp.client_id, secret)
            rows = run_sql_with_token(
                tok,
                f"SELECT COUNT(*) FROM {FQ_CRM}.leads",
                label=f"SP {sp.label} count",
            )
            n = int(rows[0][0])
        except Exception as exc:
            log(f"WARN: could not query as {sp.label}: {exc}", indent=1)
            n = -1
        counts[sp.label] = n
        log(f"{sp.label} SP sees {n} leads rows", indent=1)

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
        ref = fetch_reference_data(w)
        ensure_schema(w)
        table_counts = seed_tables(w, ref)
        ensure_policy_table(w, northstar, sunrise)
        apply_rls(w)
        apply_grants(w, [northstar, sunrise])
        total, ns_count, sr_count = verify_rls(w, northstar, sunrise)

        print()
        print("=" * 60)
        print("CRM SETUP COMPLETE")
        print("=" * 60)
        print(f"Schema: {FQ_CRM}")
        print(f"Tables: {', '.join(RLS_TABLES)}")
        for tbl, n in table_counts.items():
            print(f"  {tbl:<16} {n:>5} rows")
        print("RLS:    applied via dealership_rls() on each table")
        print(
            f"Grants: {northstar.client_id} (northstar), "
            f"{sunrise.client_id} (sunrise)"
        )
        print(
            f"Rows:   total leads={total} | "
            f"northstar SP sees {ns_count} | sunrise SP sees {sr_count}"
        )
        return 0
    except Exception as exc:
        print(f"\nFAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
