"""MLflow tracing setup.

`mlflow.langchain.autolog()` instruments every `langchain-core` Runnable at
import time — which covers LangGraph nodes, ChatDatabricks calls, tool
invocations, and our summarization LLM call. Each `graph.ainvoke(...)` becomes
one MLflow trace with nested spans per node + LLM call.

When `MLFLOW_TRACE_UC_CATALOG` + `MLFLOW_TRACE_UC_SCHEMA` are set, trace spans
are exported to a Unity Catalog schema (V4 OTEL tables). The OTEL table
prefix is `MLFLOW_TRACE_UC_PREFIX` — defaulted to `genie_sp_demo_v2` so traces
from this app don't collide with other apps sharing the schema.

Preconditions for the UC destination (per Databricks "OpenTelemetry on
Databricks" preview):
  * mlflow[databricks] >= 3.11.0
  * Preview enabled on the workspace
  * Grants on the UC schema: USE_CATALOG, USE_SCHEMA, MODIFY + SELECT on each
    of the 4 OTEL tables (ALL_PRIVILEGES is NOT sufficient)
  * `MLFLOW_TRACING_SQL_WAREHOUSE_ID` set to a warehouse the identity has
    CAN USE on (we default it to DBX_WAREHOUSE_ID)
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any, Dict, Iterator

log = logging.getLogger("genie_sp_demo_v2.tracing")

_initialized = False
_active = False  # True once MLflow is configured AND autolog succeeded


def init() -> None:
    """Enable MLflow tracing for LangChain / LangGraph. Idempotent.

    Reads settings at call time (after dotenv has loaded).
    """
    global _initialized
    if _initialized:
        return

    from .config import settings

    if not settings.mlflow_enabled:
        log.info("MLflow tracing disabled (MLFLOW_ENABLED=false)")
        _initialized = True
        return

    try:
        import mlflow
    except ImportError:
        log.warning("mlflow not installed; skipping tracing setup")
        _initialized = True
        return

    # Propagate DBX_HOST / DBX_PROFILE → standard SDK env vars so MLflow's
    # 'databricks' tracking URI can resolve credentials without requiring the
    # caller to separately set DATABRICKS_HOST / DATABRICKS_CONFIG_PROFILE.
    if settings.dbx_host and not os.environ.get("DATABRICKS_HOST"):
        os.environ["DATABRICKS_HOST"] = settings.dbx_host
    dbx_profile = os.environ.get("DBX_PROFILE", "")
    if dbx_profile and not os.environ.get("DATABRICKS_CONFIG_PROFILE"):
        os.environ["DATABRICKS_CONFIG_PROFILE"] = dbx_profile

    # SQL warehouse must be visible to MLflow before any trace write — the UC
    # trace export path reads it at module load time.
    if settings.mlflow_tracing_sql_warehouse_id:
        os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = (
            settings.mlflow_tracing_sql_warehouse_id
        )

    # 1. tracking URI + experiment (bound to UC trace location if configured).
    # UC binding is creation-time-only; if the experiment already exists without
    # it, `scripts/setup_mlflow_tracing.py` handles the delete+recreate.
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        if settings.mlflow_trace_uc_catalog and settings.mlflow_trace_uc_schema:
            from mlflow.entities.trace_location import UnityCatalog
            mlflow.set_experiment(
                experiment_name=settings.mlflow_experiment_name,
                trace_location=UnityCatalog(
                    catalog_name=settings.mlflow_trace_uc_catalog,
                    schema_name=settings.mlflow_trace_uc_schema,
                    table_prefix=settings.mlflow_trace_uc_prefix,
                ),
            )
        else:
            mlflow.set_experiment(settings.mlflow_experiment_name)
    except Exception as e:
        log.error("MLflow experiment setup failed: %s — tracing disabled", e)
        _initialized = True
        return

    # 3. autolog LangChain runnables (covers LangGraph + ChatDatabricks)
    global _active
    try:
        mlflow.langchain.autolog()
        _active = True
    except Exception as e:
        log.error("mlflow.langchain.autolog() failed: %s — tracing partially disabled", e)

    log.info(
        "MLflow tracing ready: tracking_uri=%s experiment=%s",
        settings.mlflow_tracking_uri,
        settings.mlflow_experiment_name,
    )
    _initialized = True


class _SpanHandle:
    """Tiny handle yielded by `chat_turn_span`. Lets callers set outputs on
    the root span even if tracing is disabled (then set_outputs is a no-op).
    """

    def __init__(self, span: Any | None):
        self.span = span

    def set_outputs(self, **outputs: Any) -> None:
        if self.span is None:
            return
        try:
            self.span.set_outputs(outputs)
        except Exception as e:
            log.debug("span.set_outputs failed: %s", e)


@contextlib.contextmanager
def chat_turn_span(
    *,
    thread_id: str,
    username: str,
    dealership: str,
    sp_label: str,
    user_message: str,
) -> Iterator[_SpanHandle]:
    """Root span for one /api/chat turn.

    - Tagged `mlflow.trace.session = thread_id` → MLflow Sessions UI groups
      all turns of one thread under the same session.
    - Span type `AGENT` so the trace UI renders it as a top-level agent turn.
    - `inputs` = the user's message (the one thing you want to see at the
      root); nested spans carry the intermediate reasoning.
    - Outputs are populated by the caller via `handle.set_outputs(...)`
      once the graph finishes.
    - Never logs the SP token. That's handled by not putting it in state.

    No-op if tracing is disabled or failed to initialize.
    """
    if not _active:
        yield _SpanHandle(None)
        return
    try:
        import mlflow
        from mlflow.entities import SpanType
    except ImportError:
        yield _SpanHandle(None)
        return

    try:
        with mlflow.start_span(
            name="chat_turn",
            span_type=SpanType.AGENT,
            attributes={
                "thread_id": thread_id,
                "user": username,
                "dealership": dealership,
                "sp_label": sp_label,
            },
        ) as span:
            try:
                span.set_inputs({"user_message": user_message})
            except Exception as e:
                log.debug("span.set_inputs failed: %s", e)
            try:
                mlflow.update_current_trace(
                    tags={
                        "mlflow.trace.session": thread_id,
                        "user": username,
                        "dealership": dealership,
                        "sp_label": sp_label,
                    }
                )
            except Exception as e:
                log.debug("update_current_trace failed: %s", e)
            yield _SpanHandle(span)
    except Exception as e:
        log.warning("chat_turn_span failed, continuing without wrapper: %s", e)
        yield _SpanHandle(None)
