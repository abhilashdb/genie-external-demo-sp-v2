"""Load environment variables and expose a typed Settings object.

Fails fast at import time if required vars are missing. GENIE_SPACE_ID is
allowed to be blank (Agent A populates it after the Genie space is created)
and will emit a warning instead.
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv

# Find the .env file at the project root (one level up from backend/).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=False)


class ConfigError(RuntimeError):
    """Raised when a required env var is missing."""


def _require(name: str) -> str:
    val = os.environ.get(name)
    if val is None or val.strip() == "":
        raise ConfigError(
            f"Missing required environment variable: {name} (expected in {_ENV_PATH})"
        )
    return val.strip()


def _optional(name: str, default: str = "") -> str:
    val = os.environ.get(name, default)
    return val.strip() if isinstance(val, str) else default


@dataclass(frozen=True)
class Settings:
    # Databricks workspace
    dbx_host: str
    dbx_warehouse_id: str
    dbx_catalog: str
    dbx_schema: str

    # Genie space (may be blank during initial setup)
    genie_space_id: str

    # Service principals
    sp_northstar_client_id: str
    sp_northstar_secret: str
    sp_northstar_dealership: str
    sp_sunrise_client_id: str
    sp_sunrise_secret: str
    sp_sunrise_dealership: str

    # Optional SP app ids (not in .env yet — used only in sp_mapping return)
    sp_northstar_app_id: str = ""
    sp_sunrise_app_id: str = ""

    # Session / server
    app_session_secret: str = ""
    backend_host: str = "127.0.0.1"
    backend_port: int = 8000
    frontend_port: int = 5173

    # v2: Lakebase (Postgres) URI for LangGraph checkpointer + app_conversations
    lakebase_pg_uri: str = ""

    # v2: Foundation Model API (Claude) endpoint used by the supervisor
    fm_endpoint_name: str = "databricks-claude-sonnet-4"
    fm_temperature: float = 0.2
    fm_max_tokens: int = 2048

    # v2: memory compression thresholds
    summary_turn_threshold: int = 12
    summary_token_threshold: int = 6000
    summary_keep_last: int = 4

    # v2: Genie space scope — free-form description of what the configured
    # Genie space can and cannot answer. Supervisor uses this to refuse
    # out-of-scope queries with a clear message instead of calling Genie.
    # (legacy single-space fields — kept for back-compat)
    genie_space_scope: str = ""

    # v2.1: Multi-space support. `genie_space_id` is the sales-and-service space
    # (backwards-compatible alias). `genie_space_crm_id` is an optional second
    # space for CRM / pipeline / forecast questions. The supervisor routes
    # between them based on the scope descriptions below.
    genie_space_sales_id: str = ""
    genie_space_sales_scope: str = ""
    genie_space_crm_id: str = ""
    genie_space_crm_scope: str = ""

    # v2: MLflow tracing for the LangGraph agent
    mlflow_enabled: bool = True
    mlflow_tracking_uri: str = "databricks"
    mlflow_experiment_name: str = "/Users/abhilash.r@databricks.com/genie-sp-demo-v2"
    # UC trace destination (V4 OTEL tables). Leave catalog/schema blank to use
    # the default MLflow tracking backend instead.
    mlflow_trace_uc_catalog: str = ""
    mlflow_trace_uc_schema: str = ""
    mlflow_trace_uc_prefix: str = "genie_sp_demo_v2"
    mlflow_tracing_sql_warehouse_id: str = ""

    def genie_spaces(self) -> Dict[str, Dict[str, str]]:
        """Return configured Genie spaces as {key: {id, scope, label}}.

        Keys are the tool-name suffixes ('sales', 'crm'). Spaces with empty ids
        are omitted — the supervisor only sees tools for spaces that exist.
        """
        out: Dict[str, Dict[str, str]] = {}
        if self.genie_space_sales_id:
            out["sales"] = {
                "id": self.genie_space_sales_id,
                "scope": self.genie_space_sales_scope,
                "label": "Sales & Service",
            }
        if self.genie_space_crm_id:
            out["crm"] = {
                "id": self.genie_space_crm_id,
                "scope": self.genie_space_crm_scope,
                "label": "CRM / Pipeline",
            }
        return out

    @property
    def project_root(self) -> Path:
        return _PROJECT_ROOT

    def token_url(self) -> str:
        """Databricks OAuth token endpoint for M2M client_credentials."""
        return f"{self.dbx_host.rstrip('/')}/oidc/v1/token"

    def api_base(self) -> str:
        return self.dbx_host.rstrip("/")


def load_settings() -> Settings:
    # Required (fail fast)
    dbx_host = _require("DBX_HOST")
    dbx_warehouse_id = _require("DBX_WAREHOUSE_ID")
    dbx_catalog = _require("DBX_CATALOG")
    dbx_schema = _require("DBX_SCHEMA")

    sp_northstar_client_id = _require("SP_NORTHSTAR_CLIENT_ID")
    sp_northstar_secret = _require("SP_NORTHSTAR_SECRET")
    sp_northstar_dealership = _require("SP_NORTHSTAR_DEALERSHIP")

    sp_sunrise_client_id = _require("SP_SUNRISE_CLIENT_ID")
    sp_sunrise_secret = _require("SP_SUNRISE_SECRET")
    sp_sunrise_dealership = _require("SP_SUNRISE_DEALERSHIP")

    app_session_secret = _require("APP_SESSION_SECRET")

    # Optional (warn only)
    genie_space_id = _optional("GENIE_SPACE_ID")
    if not genie_space_id:
        warnings.warn(
            "GENIE_SPACE_ID is not set; chat calls will fail until Agent A "
            "populates it in .env.",
            stacklevel=2,
        )

    backend_host = _optional("BACKEND_HOST", "127.0.0.1")
    try:
        backend_port = int(_optional("BACKEND_PORT", "8000"))
    except ValueError:
        backend_port = 8000
    try:
        frontend_port = int(_optional("FRONTEND_PORT", "5173"))
    except ValueError:
        frontend_port = 5173

    lakebase_pg_uri = _optional("LAKEBASE_PG_URI")
    if not lakebase_pg_uri or "REPLACE_ME" in lakebase_pg_uri:
        warnings.warn(
            "LAKEBASE_PG_URI not set (or placeholder). Chat will 503 until you "
            "fill it in .env.",
            stacklevel=2,
        )
        lakebase_pg_uri = ""

    def _float(name: str, default: float) -> float:
        try:
            return float(_optional(name, str(default)))
        except ValueError:
            return default

    def _int(name: str, default: int) -> int:
        try:
            return int(_optional(name, str(default)))
        except ValueError:
            return default

    return Settings(
        dbx_host=dbx_host,
        dbx_warehouse_id=dbx_warehouse_id,
        dbx_catalog=dbx_catalog,
        dbx_schema=dbx_schema,
        genie_space_id=genie_space_id,
        sp_northstar_client_id=sp_northstar_client_id,
        sp_northstar_secret=sp_northstar_secret,
        sp_northstar_dealership=sp_northstar_dealership,
        sp_sunrise_client_id=sp_sunrise_client_id,
        sp_sunrise_secret=sp_sunrise_secret,
        sp_sunrise_dealership=sp_sunrise_dealership,
        sp_northstar_app_id=_optional("SP_NORTHSTAR_APP_ID"),
        sp_sunrise_app_id=_optional("SP_SUNRISE_APP_ID"),
        app_session_secret=app_session_secret,
        backend_host=backend_host,
        backend_port=backend_port,
        frontend_port=frontend_port,
        lakebase_pg_uri=lakebase_pg_uri,
        fm_endpoint_name=_optional("FM_ENDPOINT_NAME", "databricks-claude-sonnet-4"),
        fm_temperature=_float("FM_TEMPERATURE", 0.2),
        fm_max_tokens=_int("FM_MAX_TOKENS", 2048),
        summary_turn_threshold=_int("SUMMARY_TURN_THRESHOLD", 12),
        summary_token_threshold=_int("SUMMARY_TOKEN_THRESHOLD", 6000),
        summary_keep_last=_int("SUMMARY_KEEP_LAST", 4),
        genie_space_scope=_optional("GENIE_SPACE_SCOPE"),
        # Sales space falls back to the legacy GENIE_SPACE_ID if SALES_ID not set.
        genie_space_sales_id=_optional("GENIE_SPACE_SALES_ID") or genie_space_id,
        genie_space_sales_scope=(
            _optional("GENIE_SPACE_SALES_SCOPE") or _optional("GENIE_SPACE_SCOPE")
        ),
        genie_space_crm_id=_optional("GENIE_SPACE_CRM_ID"),
        genie_space_crm_scope=_optional("GENIE_SPACE_CRM_SCOPE"),
        mlflow_enabled=_optional("MLFLOW_ENABLED", "true").lower() in ("1", "true", "yes", "on"),
        mlflow_tracking_uri=_optional("MLFLOW_TRACKING_URI", "databricks"),
        mlflow_experiment_name=_optional(
            "MLFLOW_EXPERIMENT_NAME",
            "/Users/abhilash.r@databricks.com/genie-sp-demo-v2",
        ),
        mlflow_trace_uc_catalog=_optional("MLFLOW_TRACE_UC_CATALOG"),
        mlflow_trace_uc_schema=_optional("MLFLOW_TRACE_UC_SCHEMA"),
        mlflow_trace_uc_prefix=_optional("MLFLOW_TRACE_UC_PREFIX", "genie_sp_demo_v2"),
        mlflow_tracing_sql_warehouse_id=(
            _optional("MLFLOW_TRACING_SQL_WAREHOUSE_ID") or dbx_warehouse_id
        ),
    )


# Import-time load — raises if required vars are missing.
settings: Settings = load_settings()
