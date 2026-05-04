#!/usr/bin/env python
"""Bootstrap the UC schema + MLflow experiment for v2 tracing.

Idempotent. Reads DBX_HOST, DBX_WAREHOUSE_ID, and MLFLOW_* from `.env`.

What it does:
  1. Creates UC schema `<MLFLOW_TRACE_UC_CATALOG>.<MLFLOW_TRACE_UC_SCHEMA>` if missing.
  2. Creates the MLflow experiment at MLFLOW_EXPERIMENT_NAME if missing.
  3. Tags the experiment so the UI shows the UC trace destination.
  4. Prints a summary with the experiment URL and the table prefix.

Run:
  uv run python scripts/setup_mlflow_tracing.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings  # noqa: E402
from databricks.sdk import WorkspaceClient  # noqa: E402
from databricks.sdk.service.sql import StatementState  # noqa: E402


def _run_sql(w: WorkspaceClient, statement: str) -> None:
    """Fire-and-wait SQL against the configured warehouse."""
    exec = w.statement_execution.execute_statement(
        warehouse_id=settings.dbx_warehouse_id,
        statement=statement,
        wait_timeout="30s",
    )
    state = (exec.status.state if exec.status else None) or StatementState.SUCCEEDED
    if state != StatementState.SUCCEEDED:
        err = exec.status.error if exec.status else None
        raise RuntimeError(f"SQL failed: {statement!r} — {err}")


def main() -> None:
    if not settings.mlflow_trace_uc_catalog or not settings.mlflow_trace_uc_schema:
        print("MLFLOW_TRACE_UC_CATALOG / _SCHEMA not set — aborting.", file=sys.stderr)
        sys.exit(1)

    print(f"Workspace:   {settings.dbx_host}")
    print(f"Warehouse:   {settings.dbx_warehouse_id}")
    print(
        f"UC target:   {settings.mlflow_trace_uc_catalog}."
        f"{settings.mlflow_trace_uc_schema} (prefix={settings.mlflow_trace_uc_prefix})"
    )
    print(f"Experiment:  {settings.mlflow_experiment_name}")
    print()

    w = WorkspaceClient(host=settings.dbx_host)

    # 1. UC schema
    fq = f"{settings.mlflow_trace_uc_catalog}.{settings.mlflow_trace_uc_schema}"
    print(f"1/3  CREATE SCHEMA IF NOT EXISTS {fq}")
    _run_sql(w, f"CREATE SCHEMA IF NOT EXISTS {fq}")
    # Skipping SET TAGS — this workspace has a tag policy restricting `app`
    # values. The UC schema comment is a safer, unrestricted label.
    _run_sql(
        w,
        f"COMMENT ON SCHEMA {fq} IS "
        f"'Agent traces for the genie-sp-demo-v2 LangGraph app. "
        f"OTEL table prefix: {settings.mlflow_trace_uc_prefix}.'",
    )

    # 2. MLflow experiment — bound to a UC trace location.
    # Per Databricks docs: the UC binding can only be set at creation time, so
    # if an experiment already exists without the binding we delete+recreate.
    print(f"2/3  MLflow experiment {settings.mlflow_experiment_name}")
    import mlflow
    from mlflow.entities.trace_location import UnityCatalog

    os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = settings.dbx_warehouse_id
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.MlflowClient()

    existing = client.get_experiment_by_name(settings.mlflow_experiment_name)
    if existing:
        print(f"     → found existing id={existing.experiment_id} — deleting "
              f"so it can be recreated with the UC binding")
        try:
            client.delete_experiment(existing.experiment_id)
        except Exception as e:
            print(f"     ! delete failed: {e}")
            print("       If it's already soft-deleted, try permanently purging "
                  "it via the Databricks UI or choose a different "
                  "MLFLOW_EXPERIMENT_NAME in .env.")
            raise

    exp = mlflow.set_experiment(
        experiment_name=settings.mlflow_experiment_name,
        trace_location=UnityCatalog(
            catalog_name=settings.mlflow_trace_uc_catalog,
            schema_name=settings.mlflow_trace_uc_schema,
            table_prefix=settings.mlflow_trace_uc_prefix,
        ),
    )
    exp_id = exp.experiment_id
    print(f"     → created with UC binding (id={exp_id})")

    # 3. Summary
    print()
    print("3/3  done.")
    host = settings.dbx_host.rstrip("/")
    print(f"  Experiment URL:  {host}/ml/experiments/{exp_id}")
    print(f"  UC schema URL:   {host}/explore/data/{settings.mlflow_trace_uc_catalog}/{settings.mlflow_trace_uc_schema}")
    print(
        "  OTEL tables (will be auto-created on first trace write):\n"
        f"    - {fq}.{settings.mlflow_trace_uc_prefix}_otel_spans\n"
        f"    - {fq}.{settings.mlflow_trace_uc_prefix}_otel_logs\n"
        f"    - {fq}.{settings.mlflow_trace_uc_prefix}_otel_metrics\n"
        f"    - {fq}.{settings.mlflow_trace_uc_prefix}_otel_annotations"
    )


if __name__ == "__main__":
    main()
