"""Graph state schema.

One checkpointed record per thread_id. LangGraph's MessagesState would work,
but we need extra fields (summary, genie_conv_id, auth context), so we roll
our own TypedDict.
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    # Conversation messages. `add_messages` reducer merges by id, so nodes can
    # return either full history or just the new deltas.
    messages: Annotated[List[Any], add_messages]

    # Rolling summary of messages that have been compressed out of `messages`.
    summary: str

    # Sticky Genie conv id per space, e.g. {"sales": "...", "crm": "..."}.
    # Rotated only on explicit user reset (Option D). Kept as a dict so adding
    # a space doesn't require a state migration.
    genie_conv_ids: Dict[str, str]

    # DEPRECATED — preserved for checkpoints written before multi-space support.
    # Treated as the sales conv id if `genie_conv_ids` is missing.
    genie_conv_id: Optional[str]

    # Genie results surfaced to the HTTP layer for the CURRENT turn. Cleared
    # each turn by `load_context_node`; populated by `genie_node` with one
    # entry per tool call the supervisor issued (could be multiple spaces
    # in a single turn).
    last_genie_results: List[Dict[str, Any]]

    # Append-only history of structured Genie results keyed by tool_call_id.
    # Used on transcript replay to re-attach sql/rows/columns to the right
    # assistant bubble, since the ToolMessage we send to the LLM no longer
    # carries them (we strip table+SQL to keep the LLM from echoing them).
    genie_results: Annotated[List[Dict[str, Any]], add]

    # Append-only set (encoded as a list) of tool_call_ids whose sql/rows/columns
    # have been "cleared" by a Genie reset. On replay, these bubbles render with
    # a "cleared as part of reset" note instead of the SQL+table. Append-only
    # means additional resets just add more ids to the set — duplicates are fine.
    genie_cleared_tool_call_ids: Annotated[List[str], add]

    # Auth / tenancy context carried across turns.
    local_user: str
    sp_label: str
    dealership: str

    # NOTE: `sp_token` and `session_id` are NOT in state anymore. They are
    # secrets / ephemeral scratch that must not end up in MLflow traces or
    # checkpoints. See `backend/request_ctx.py` — set at /api/chat boundary,
    # read from inside nodes via ContextVars.
