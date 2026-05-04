# Genie External SP Demo — v2

A dealership chat app that sits on top of Databricks Genie. A **LangGraph
supervisor agent** (Claude via the Databricks Foundation Model API) decides
per turn whether to route to Genie or reply directly, and checkpoints every
turn to **Lakebase** (Postgres). Two Genie spaces — Sales & Service and CRM —
are wired in; the supervisor routes by scope and can call both in one turn.

End users don't have Databricks identities; the app maps each local user to
a tenant-scoped **Service Principal**, swaps credentials via OAuth M2M, and
all Genie queries run as that SP so Unity Catalog row-level security applies
per dealership.

Everything is traced with **MLflow** into UC OTEL tables and grouped by
thread via MLflow Sessions.

## Architecture at a glance

```
Browser  →  FastAPI  →  LangGraph agent (Claude FM API)
                              ├─→ ask_sales_genie  →  Genie Space A  →  UC (sales)
                              └─→ ask_crm_genie    →  Genie Space B  →  UC (CRM)
                         ↕
                     Lakebase (Postgres)
                         ├── checkpoints           (LangGraph state per thread)
                         └── app_conversations     (sidebar pointer rows)

MLflow Traces → UC (abhilash_r.agent_tracing.genie_sp_demo_v2_otel_*)
```

Graph node layout:

```
START → load_context → maybe_summarize → supervisor ─┬─→ genie → supervisor
                                                      └─→ END
```

## Key behaviors

- **Multi-space routing with two Genie tools.** `ask_sales_genie` for
  completed sales / service data; `ask_crm_genie` for leads / pipeline /
  forecasts. Tools are only bound if the corresponding space id is set in
  `.env`, so you can run with one space during setup.
- **Verbatim pass-through.** The supervisor forwards the user's question
  to Genie as-is. It doesn't invent metrics, columns, or time windows.
- **Scope discipline.** Each space's scope string is in `.env`. Out-of-scope
  questions get a clean refusal. Ambiguous words (`deals`, `pipeline`,
  `leads`, `opportunity`, `forecast`, …) trigger a clarify-first step so the
  user picks the intended space.
- **No RCA.** Genie's agent-mode RCA isn't API-exposed. "Why" / "how can I
  improve" questions get turned into concrete factual data cuts the user
  can inspect themselves.
- **Rolling-summary memory.** When a thread exceeds
  `SUMMARY_TURN_THRESHOLD` turns or `SUMMARY_TOKEN_THRESHOLD` tokens, a
  summarizer node folds older messages into a `summary` string and keeps
  the last `SUMMARY_KEEP_LAST` verbatim.
- **Per-space sticky Genie convs.** Each space has its own sticky
  `genie_conv_id` so Genie's native follow-up context is preserved within
  a space. A "Reset data analysis (Genie)" button rotates the conv for
  one (or all) spaces; pre-reset tool results are marked *cleared as part
  of reset* on replay.
- **Structured traces.** Each `/api/chat` turn is wrapped in an MLflow
  root span (`span_type=AGENT`) with the user's message as input and the
  final reply + spaces used as output. All intermediate reasoning (LLM
  calls, Genie calls, summarization) nests as child spans. Traces are
  tagged `mlflow.trace.session = thread_id` so the MLflow Sessions UI
  groups a whole conversation.
- **Secrets stay out of state.** SP tokens and session ids are passed via
  `asyncio.ContextVar` (see `backend/request_ctx.py`), never placed into
  the LangGraph state dict — so they never appear in checkpoints or traces.

## Setup

### 0. Prerequisites

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) (`brew install uv`)
- Databricks CLI authenticated against a workspace where you have CREATEDB
  on a Lakebase instance and CREATE SCHEMA on a UC catalog
- A Lakebase instance reachable from your machine
- A serving endpoint for Claude (defaults to `databricks-claude-sonnet-4`)

### 1. Seed env

```bash
cp .env.example .env
```

Fill in:

- `DBX_HOST`, `DBX_WAREHOUSE_ID`, `DBX_CATALOG`, `DBX_SCHEMA`
- `SP_NORTHSTAR_*` and `SP_SUNRISE_*` — service principal credentials. Both
  SPs need CAN_RUN on both Genie spaces and the right grants on the UC
  tables. `setup_databricks.py` + `setup_crm_databricks.py` handle the
  grants for you.
- `APP_SESSION_SECRET` — random 48+ char string to sign session cookies.
- `LAKEBASE_PG_URI` — `postgresql://<user>:<pw>@<host>:5432/<db>?sslmode=require`.
- `FM_ENDPOINT_NAME` — defaults to `databricks-claude-sonnet-4`.
- `MLFLOW_EXPERIMENT_NAME` — an absolute workspace path you own.
- `MLFLOW_TRACE_UC_CATALOG` / `_SCHEMA` / `_PREFIX` — for UC OTEL trace storage.
- `GENIE_SPACE_SALES_ID` / `GENIE_SPACE_CRM_ID` — filled in by the seed scripts
  (step 3).

### 2. Install dependencies

```bash
uv sync
```

### 3. Seed Databricks (idempotent, safe to re-run)

Each script prints a line you paste into `.env` when it's done. Run them in
this order:

```bash
# Sales & Service domain: UC schema + tables + RLS + Genie space
uv run python scripts/setup_databricks.py
uv run python scripts/create_genie_space.py
#   → GENIE_SPACE_SALES_ID=…

# CRM domain: UC schema + tables + RLS + Genie space (references the sales
# data so salesperson names / dealerships reconcile across both spaces)
uv run python scripts/setup_crm_databricks.py
uv run python scripts/create_crm_genie_space.py
#   → GENIE_SPACE_CRM_ID=…

# MLflow experiment bound to the UC trace schema
uv run python scripts/setup_mlflow_tracing.py
```

The MLflow setup script creates `abhilash_r.agent_tracing` (if missing) and
creates the experiment with `trace_location=UnityCatalog(...)` — binding is
only possible at creation time, so if an experiment of the same name exists
without the binding the script deletes + recreates it.

### 4. Grant the Lakebase role CREATE on public

The app auto-creates checkpoint tables + `app_conversations`. The DB role in
`LAKEBASE_PG_URI` needs `CREATE` on `public`:

```sql
GRANT CREATE ON SCHEMA public TO "<role_name>";
```

### 5. Run

```bash
uv run python -m backend.run
# -> http://127.0.0.1:8002
```

Log in as one of the seed users. All passwords are `demo123`:

| Username | Dealership | Role | Maps to SP |
|---|---|---|---|
| alice | North Star Motors | manager | northstar |
| bob | North Star Motors | analyst | northstar |
| carol | Sunrise Auto Group | manager | sunrise |
| dave | Sunrise Auto Group | analyst | sunrise |

## Demo script

Once signed in:

1. **Single-space, verbatim forward** — "what were my top salespeople last
   month?" → routes to Sales space, renders SQL + table with an indigo
   "Answered via Sales & Service" badge.
2. **Single-space, CRM** — "which of my leads haven't had activity in 30+
   days?" → routes to CRM, pink badge.
3. **Ambiguity disambiguation** — "show me my deals" → agent asks a/b
   (completed sales vs. pipeline) instead of guessing.
4. **Out-of-scope refusal** — "what's my marketing spend by campaign?" →
   agent refuses with a specific list of what it *can* answer.
5. **Multi-space in one turn** — "show my completed sales this quarter and
   my open pipeline" → two badges + two result blocks in a single bubble.
6. **Memory compression** — lower `SUMMARY_TURN_THRESHOLD` to 4 in `.env`,
   send 5+ turns, watch the `agent_summarize` event fire and the Memory
   panel update with a rolling summary.
7. **Reset** — click "Reset data analysis (Genie)"; prior results show the
   *cleared as part of reset* marker; agent summary is preserved.
8. **Observability** — click "View traces ↗" in the top bar to open the
   MLflow experiment in a new tab. Sessions tab groups all turns of a
   thread together.

## API

| Method | Path | Notes |
|---|---|---|
| `POST` | `/api/login` | session cookie auth |
| `POST` | `/api/logout` | |
| `GET`  | `/api/me` | |
| `POST` | `/api/chat` | `{message, thread_id?}` → `{thread_id, answer_text, results: [{space, space_label, sql, rows, columns}], summary}` |
| `POST` | `/api/threads/{thread_id}/reset-genie` | optional body `{space: "sales"|"crm"}`; omit to reset all |
| `GET`  | `/api/conversations` | sidebar list |
| `GET`  | `/api/conversations/{thread_id}` | full transcript + summary; falls back to Genie list-messages for turns predating the `genie_results` field |
| `GET`  | `/api/events/stream` | SSE transparency pane |
| `GET`  | `/api/tracing/experiment` | MLflow experiment URL for the "View traces" link |
| `GET`  | `/api/health` | readiness |

## Code layout

```
backend/
  main.py             — FastAPI, /api/chat, reset, thread listing
  config.py           — typed settings from .env
  lakebase.py         — psycopg pool + AsyncPostgresSaver + app_conversations
  tracing.py          — MLflow autolog + chat_turn_span (AGENT root span)
  request_ctx.py      — asyncio ContextVar for sp_token/session_id (never traced)
  flow_events.py      — SSE bus (UI transparency pane)
  databricks_auth.py  — OAuth M2M token exchange, cached per sp_label
  genie_client.py     — async httpx wrapper around Genie REST, retries, SSE
  sp_mapping.py       — sp_label → (client_id, secret, dealership, app_id)
  users.py            — seeded local users (demo only)
  agent/
    state.py          — AgentState TypedDict
    graph.py          — StateGraph; supervisor + genie_node + tools
    tokens.py         — tiktoken-based counter for compression triggers
  static/             — single-page UI (no build step)

scripts/
  setup_databricks.py         — UC schema + Sales tables + RLS + grants
  create_genie_space.py       — Sales Genie space + sample questions
  setup_crm_databricks.py     — UC schema + CRM tables + RLS + grants
  create_crm_genie_space.py   — CRM Genie space + sample questions
  setup_mlflow_tracing.py     — UC agent_tracing schema + MLflow experiment (UC-bound)
  debug_genie_result.py       — poke Genie's query-result for a conv_id + msg_id
  teardown_databricks.py      — drop everything we created (use with care)
```

## Troubleshooting

- **Chat 503 "LAKEBASE_PG_URI not configured"** — fill in the URI. Check
  `/api/health`.
- **Chat 503 "GENIE_SPACE_ID not configured"** — set at least
  `GENIE_SPACE_SALES_ID`. The CRM space is optional.
- **Pool connect fails with `error connecting in 'pool-1'`** — psycopg_pool
  3.2 requires `min_size`. Already set to 1 in `lakebase.py`. If it still
  fails, the real error is upstream — confirm the URI is reachable via
  `psql "$LAKEBASE_PG_URI"`.
- **`permission denied for schema public`** on first start — the DB role
  needs `GRANT CREATE ON SCHEMA public`. See step 4.
- **MLflow autolog "unexpected keyword 'log_models'"** — you're on an older
  MLflow. `uv sync` should pull `mlflow>=3.11.1`. The tracing module catches
  this and continues without autolog.
- **UC OTEL tables stay empty** — confirm the experiment was created with
  the UC binding (the setup script prints that). UC binding is creation-time
  only; delete and recreate if needed.
- **Summary never fires** — thresholds are deliberately high. Lower
  `SUMMARY_TURN_THRESHOLD` to 2–3 for a quick demo.
