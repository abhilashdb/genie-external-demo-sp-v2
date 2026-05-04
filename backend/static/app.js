// Genie Space SP Demo — frontend
// Single-page app: login view <-> chat view. No build step, no deps.

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const state = {
  view: "login", // "login" | "chat"
  tab: "chat", // "chat" | "arch"
  user: null, // { username, dealership, role, sp_label }
  threadId: null, // v2: app thread id (scopes checkpointed state)
  genieConvId: null, // sticky Genie conv id on this thread (Option D)
  summary: "", // rolling memory summary
  turnCount: 0, // non-system messages in current thread
  messages: [], // { role: "user"|"assistant"|"system"|"tool", text, sql?, rows?, columns? }
  conversations: [], // [{ thread_id, title, genie_conv_id, last_active_at, ... }]
  historyLoading: false,
  flowEvents: [], // newest first
  eventSource: null,
  reconnectAttempted: false,
  mermaidInitialized: false,
  mermaidRendered: false,
};

// Color classes per flow step (CSS has matching .step-<name> rules)
const STEP_COLORS = {
  login: "blue",
  sp_resolve: "purple",
  token_exchange: "amber",
  agent_load_context: "indigo",
  agent_memory: "indigo",
  agent_summarize: "fuchsia",
  agent_supervisor: "violet",
  agent_reset_genie: "fuchsia",
  genie_call: "green",
  genie_rate_limit: "orange",
  genie_retry: "orange",
  genie_sql: "teal",
  sql_execute: "slate",
  rls_applied: "rose",
  response: "emerald",
  error: "red",
};

// ---------------------------------------------------------------------------
// DOM helpers
// ---------------------------------------------------------------------------
const $ = (sel) => document.querySelector(sel);

function showView(view) {
  state.view = view;
  $("#view-login").hidden = view !== "login";
  $("#view-chat").hidden = view !== "chat";
  // Topbar is auth-gated; tab 1 label reflects auth state.
  const authed = view === "chat";
  $("#topbar").hidden = !authed;
  $("#tab-main-btn").textContent = authed ? "Chat" : "Sign in";
}

function escapeHtml(s) {
  if (s === null || s === undefined) return "";
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function renderNewlines(text) {
  // Escape first, then apply a tiny subset of markdown on the escaped string
  // so we never render untrusted HTML. Supports **bold**, *italic*, `code`,
  // blank-line paragraphs, and single newlines as <br/>.
  let s = escapeHtml(text || "");
  // Inline code: `foo`  (do before bold/italic so backticks protect their content)
  s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
  // Bold: **foo**  (non-greedy, no line breaks inside)
  s = s.replace(/\*\*([^*\n]+?)\*\*/g, "<strong>$1</strong>");
  // Italic: *foo*  (single stars, not inside words)
  s = s.replace(/(^|[^*])\*([^*\n]+?)\*(?!\*)/g, "$1<em>$2</em>");
  const paragraphs = s.split(/\n{2,}/);
  return paragraphs.map((p) => `<p>${p.replace(/\n/g, "<br/>")}</p>`).join("");
}

function relativeTime(iso) {
  if (!iso) return "";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";
  const diff = Math.max(0, Date.now() - then);
  const s = Math.floor(diff / 1000);
  if (s < 5) return "just now";
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.floor(h / 24);
  return `${d}d ago`;
}

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------
async function api(path, opts = {}) {
  const resp = await fetch(path, {
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
    ...opts,
  });
  if (resp.status === 401 && state.view !== "login") {
    // Session expired mid-session — force back to login.
    teardownChat();
    showView("login");
    showLoginError("Session expired. Please sign in again.");
    throw new Error("unauthorized");
  }
  return resp;
}

function showLoginError(msg) {
  const el = $("#login-error");
  el.textContent = msg;
  el.hidden = !msg;
}

// ---------------------------------------------------------------------------
// Login / logout
// ---------------------------------------------------------------------------
async function tryRestoreSession() {
  try {
    const resp = await fetch("/api/me", { credentials: "same-origin" });
    if (resp.ok) {
      const user = await resp.json();
      enterChat(user);
      return;
    }
  } catch (_) {}
  showView("login");
}

async function handleLogin(ev) {
  ev.preventDefault();
  showLoginError("");
  const username = $("#login-username").value.trim();
  const password = $("#login-password").value;
  const submitBtn = $("#login-submit");
  submitBtn.disabled = true;
  submitBtn.textContent = "Signing in...";
  try {
    const resp = await api("/api/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: "login failed" }));
      showLoginError(err.detail || "Login failed");
      return;
    }
    const user = await resp.json();
    enterChat(user);
  } catch (e) {
    if (e.message !== "unauthorized") showLoginError("Network error. Try again.");
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Sign in";
  }
}

async function handleLogout() {
  try {
    await api("/api/logout", { method: "POST" });
  } catch (_) {}
  teardownChat();
  showView("login");
  $("#login-username").value = "";
  $("#login-password").value = "";
  showLoginError("");
}

// ---------------------------------------------------------------------------
// Chat view lifecycle
// ---------------------------------------------------------------------------
function enterChat(user) {
  state.user = user;
  state.threadId = null;
  state.genieConvId = null;
  state.summary = "";
  state.turnCount = 0;
  state.messages = [];
  state.conversations = [];
  state.flowEvents = [];
  state.reconnectAttempted = false;
  renderTopbar();
  renderMessages();
  renderHistory();
  renderFlow();
  renderMemoryPanel();
  showView("chat");
  switchTab("main");
  openEventStream();
  refreshConversations();
  refreshTracingLink();
  $("#chat-text").focus();
}

async function refreshTracingLink() {
  const el = $("#tracing-link");
  if (!el) return;
  try {
    const resp = await api("/api/tracing/experiment");
    if (!resp.ok) return;
    const data = await resp.json();
    if (data.enabled && data.url) {
      el.href = data.url;
      el.title = `MLflow experiment: ${data.experiment_name}` +
        (data.uc_schema ? ` • UC: ${data.uc_schema} (prefix ${data.uc_table_prefix})` : "");
      el.hidden = false;
    } else {
      el.hidden = true;
    }
  } catch (_) {
    // network/auth error — leave link hidden
  }
}

function teardownChat() {
  if (state.eventSource) {
    try {
      state.eventSource.close();
    } catch (_) {}
    state.eventSource = null;
  }
  state.user = null;
  state.threadId = null;
  state.genieConvId = null;
  state.summary = "";
  state.turnCount = 0;
  state.messages = [];
  state.conversations = [];
  state.flowEvents = [];
  renderMemoryPanel();
}

function renderTopbar() {
  const u = state.user;
  if (!u) return;
  $("#user-greeting").textContent = `Logged in as ${u.username} • ${u.dealership} • ${u.role}`;
  $("#sp-badge").textContent = `Connected via SP: ${u.sp_label}-sp`;
}

// ---------------------------------------------------------------------------
// History sidebar
// ---------------------------------------------------------------------------
async function refreshConversations() {
  try {
    const resp = await api("/api/conversations");
    if (!resp.ok) return;
    const data = await resp.json();
    state.conversations = data.conversations || [];
    renderHistory();
  } catch (_) {}
}

function renderHistory() {
  const container = $("#history-list");
  if (!container) return;
  if (state.conversations.length === 0) {
    container.innerHTML = `<div class="empty-state-sm">No past conversations yet.</div>`;
    return;
  }
  container.innerHTML = state.conversations
    .map((c) => {
      const active = c.thread_id === state.threadId ? " active" : "";
      const title = escapeHtml(c.title || "(untitled)");
      return `
        <button type="button" class="history-item${active}" data-thread-id="${escapeHtml(
          c.thread_id,
        )}" title="${escapeHtml(c.last_active_at || "")}">
          <div class="history-title">${title}</div>
          <div class="history-meta">${escapeHtml(relativeTime(c.last_active_at))}</div>
        </button>`;
    })
    .join("");
  container.querySelectorAll(".history-item").forEach((el) => {
    el.addEventListener("click", () => handleOpenConversation(el.dataset.threadId));
  });
}

async function handleOpenConversation(threadId) {
  if (!threadId || state.historyLoading) return;
  if (threadId === state.threadId) return;
  state.historyLoading = true;
  state.threadId = threadId;
  state.messages = [{ role: "system", text: "Loading conversation..." }];
  renderMessages();
  renderHistory();
  try {
    const resp = await api(`/api/conversations/${encodeURIComponent(threadId)}`);
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: `HTTP ${resp.status}` }));
      state.messages = [{ role: "system", text: `Error: ${err.detail || "failed to load"}` }];
      renderMessages();
      return;
    }
    const data = await resp.json();
    state.genieConvId = data.genie_conv_id || null;
    state.summary = data.summary || "";
    state.messages = (data.messages || []).map((m) => ({
      role: m.role,
      text: m.text,
      results: m.results || [],
      cleared: !!m.cleared,
    }));
    state.turnCount = state.messages.filter((m) => m.role !== "system").length;
    renderMessages();
    renderMemoryPanel();
  } catch (e) {
    if (e.message !== "unauthorized") {
      state.messages = [{ role: "system", text: "Network error loading conversation." }];
      renderMessages();
    }
  } finally {
    state.historyLoading = false;
  }
}

function handleNewChat() {
  state.threadId = null;
  state.genieConvId = null;
  state.summary = "";
  state.turnCount = 0;
  state.messages = [];
  renderMessages();
  renderHistory();
  renderMemoryPanel();
  $("#chat-text").focus();
}

async function handleResetGenie() {
  if (!state.threadId) return;
  const btn = $("#reset-genie-btn");
  btn.disabled = true;
  const prev = btn.textContent;
  btn.textContent = "Resetting…";
  try {
    const resp = await api(`/api/threads/${encodeURIComponent(state.threadId)}/reset-genie`, {
      method: "POST",
    });
    if (resp.ok) {
      state.genieConvId = null;
      state.messages.push({
        role: "system",
        text: "New topic — Genie will start a fresh conversation on the next data question. Agent summary preserved.",
      });
      renderMessages();
      renderMemoryPanel();
    }
  } catch (_) {
    // next chat turn will surface the real error
  } finally {
    btn.disabled = false;
    btn.textContent = prev;
  }
}

function renderMemoryPanel() {
  const panel = $("#memory-panel");
  if (!panel) return;
  if (!state.threadId) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  const stats = `${state.turnCount} turn${state.turnCount === 1 ? "" : "s"}${
    state.genieConvId ? " · Genie conv active" : " · no Genie conv yet"
  }`;
  $("#memory-stats").textContent = stats;
  $("#memory-summary").textContent = state.summary
    ? state.summary
    : "(empty — first few turns are kept verbatim)";
}

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------
function pushMessage(msg) {
  state.messages.push(msg);
  renderMessages();
}

function renderMessages() {
  const container = $("#messages");
  if (state.messages.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <h3>Ask Genie about your dealership</h3>
        <p>Try: "What were my top selling vehicles last quarter?"</p>
      </div>`;
    return;
  }
  container.innerHTML = state.messages.map(renderBubble).join("");
  container.scrollTop = container.scrollHeight;
}

function renderBubble(msg) {
  if (msg.role === "system") {
    return `<div class="bubble bubble-system"><div class="bubble-body">${escapeHtml(msg.text)}</div></div>`;
  }
  if (msg.role === "tool") {
    const name = msg.name ? `<strong>${escapeHtml(msg.name)}</strong>: ` : "";
    return `<div class="bubble bubble-tool"><div class="bubble-body">${name}${renderNewlines(msg.text)}</div></div>`;
  }
  if (msg.role === "user") {
    return `<div class="bubble bubble-user"><div class="bubble-body">${renderNewlines(msg.text)}</div></div>`;
  }
  // assistant
  if (msg.loading) {
    const label = escapeHtml(msg.loadingText || "Thinking…");
    return `<div class="bubble bubble-assistant"><div class="bubble-body"><span class="loader"></span> ${label}</div></div>`;
  }
  let body = "";
  const results = Array.isArray(msg.results) ? msg.results : [];
  if (results.length > 0) {
    body += `<div class="space-badges">`;
    for (const r of results) {
      const label = r.space_label || r.spaceLabel || "Genie";
      const key = r.space || "";
      body += `<span class="space-badge space-${escapeHtml(key)}">Answered via <strong>${escapeHtml(label)}</strong> Genie space</span>`;
    }
    body += `</div>`;
  }
  if (msg.text) body += `<div class="answer-text">${renderNewlines(msg.text)}</div>`;
  if (msg.cleared && results.length === 0) {
    body += `<div class="cleared-note">SQL and results cleared as part of Genie reset.</div>`;
  } else if (results.length > 0) {
    for (const r of results) {
      body += renderResultBlock(r);
    }
  }
  return `<div class="bubble bubble-assistant"><div class="bubble-body">${body}</div></div>`;
}

function renderResultBlock(r) {
  const label = r.space_label || r.spaceLabel || "";
  const sql = r.sql || null;
  const rows = r.rows || null;
  const columns = r.columns || null;
  let html = `<div class="result-block">`;
  if (label) html += `<div class="result-block-label">${escapeHtml(label)}</div>`;
  if (sql) {
    html += `
      <details class="sql-block">
        <summary>Show SQL</summary>
        <pre><code>${escapeHtml(sql)}</code></pre>
      </details>`;
  }
  if (rows && columns && rows.length > 0) {
    html += renderTable(columns, rows);
  } else if (rows && rows.length === 0) {
    html += `<div class="rows-empty">(query returned 0 rows)</div>`;
  }
  html += `</div>`;
  return html;
}

function renderTable(columns, rows) {
  const head = columns
    .map((c) => `<th>${escapeHtml(c.name)}<span class="col-type">${escapeHtml(c.type || "")}</span></th>`)
    .join("");
  const shown = rows.slice(0, 50);
  const body = shown
    .map((row) => {
      const cells = columns
        .map((c) => {
          const v = row[c.name];
          const display = v === null || v === undefined ? "" : typeof v === "object" ? JSON.stringify(v) : String(v);
          return `<td>${escapeHtml(display)}</td>`;
        })
        .join("");
      return `<tr>${cells}</tr>`;
    })
    .join("");
  const footer =
    rows.length > 50 ? `<div class="table-footnote">Showing 50 of ${rows.length} rows.</div>` : "";
  return `<div class="result-table-wrap"><table class="result-table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>${footer}</div>`;
}

async function handleChatSubmit(ev) {
  ev.preventDefault();
  const input = $("#chat-text");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";

  pushMessage({ role: "user", text });
  state.turnCount += 1;
  const placeholder = { role: "assistant", loading: true, loadingText: "Thinking…" };
  state.pendingPlaceholder = placeholder;
  pushMessage(placeholder);
  renderMemoryPanel();

  try {
    const resp = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ message: text, thread_id: state.threadId }),
    });
    state.messages = state.messages.filter((m) => m !== placeholder);
    state.pendingPlaceholder = null;
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: `HTTP ${resp.status}` }));
      pushMessage({ role: "system", text: `Error: ${err.detail || "request failed"}` });
      return;
    }
    const data = await resp.json();
    const isNewThread = !state.threadId && !!data.thread_id;
    if (data.thread_id) state.threadId = data.thread_id;
    state.genieConvId = data.genie_conv_id || state.genieConvId;
    state.summary = data.summary || state.summary;
    pushMessage({
      role: "assistant",
      text: data.answer_text,
      results: data.results || [],
    });
    state.turnCount += 1;
    renderMemoryPanel();
    refreshConversations();
    if (isNewThread) renderHistory();
  } catch (e) {
    state.messages = state.messages.filter((m) => m !== placeholder);
    state.pendingPlaceholder = null;
    if (e.message !== "unauthorized") {
      pushMessage({ role: "system", text: "Network error contacting backend." });
    }
    renderMessages();
  }
}

// ---------------------------------------------------------------------------
// Flow events (SSE)
// ---------------------------------------------------------------------------
function openEventStream() {
  if (state.eventSource) {
    try { state.eventSource.close(); } catch (_) {}
  }
  const es = new EventSource("/api/events/stream", { withCredentials: true });
  state.eventSource = es;

  es.addEventListener("flow", (e) => {
    try {
      const payload = JSON.parse(e.data);
      state.flowEvents.unshift(payload);
      renderFlow();
      updatePlaceholderForStep(payload);
    } catch (err) {
      // swallow parse errors
    }
  });

  es.addEventListener("error", () => {
    // Try one reconnect after 2s, then give up (browser will also auto-retry EventSource).
    if (state.reconnectAttempted) return;
    state.reconnectAttempted = true;
    setTimeout(() => {
      if (state.view === "chat") openEventStream();
    }, 2000);
  });
}

// Map SSE steps to the live placeholder label. Most specific wins: as soon as
// we know the agent called Genie, we show "Querying data…"; otherwise the
// label stays at whatever higher-level phase we're in.
const PLACEHOLDER_LABELS = {
  agent_load_context: "Loading thread memory…",
  agent_memory: "Checking memory usage…",
  agent_summarize: "Compressing conversation memory…",
  agent_supervisor: "Thinking…",
  token_exchange: "Authenticating with Databricks…",
  genie_call: "Querying data (Genie)…",
  genie_rate_limit: "Genie rate-limited — backing off…",
  genie_retry: "Retrying data query…",
  genie_sql: "Running SQL on your data…",
  sql_execute: "Running SQL on your data…",
};

function updatePlaceholderForStep(ev) {
  const p = state.pendingPlaceholder;
  if (!p || !p.loading) return;
  const label = PLACEHOLDER_LABELS[ev.step];
  if (!label) return;
  if (p.loadingText === label) return;
  p.loadingText = label;
  renderMessages();
}

function renderFlow() {
  const container = $("#flow-events");
  if (state.flowEvents.length === 0) {
    container.innerHTML = `<div class="empty-state-sm">No flow events yet. Send a message to see the backend trace.</div>`;
    return;
  }
  container.innerHTML = state.flowEvents.map(renderFlowCard).join("");
}

function renderFlowCard(ev) {
  const color = STEP_COLORS[ev.step] || "slate";
  const status = ev.status || "ok";
  const payloadJson = ev.payload ? JSON.stringify(ev.payload, null, 2) : "";
  const detail = ev.detail ? `<div class="flow-detail">${escapeHtml(ev.detail)}</div>` : "";
  const payloadBlock = payloadJson
    ? `<details class="flow-payload"><summary>payload</summary><pre>${escapeHtml(payloadJson)}</pre></details>`
    : "";
  return `
    <article class="flow-card step-${color} status-${status}">
      <div class="flow-card-head">
        <span class="step-pill step-pill-${color}">${escapeHtml(ev.step || "")}</span>
        <time class="flow-ts" data-ts="${escapeHtml(ev.ts || "")}">${escapeHtml(relativeTime(ev.ts))}</time>
      </div>
      <div class="flow-title">${escapeHtml(ev.title || "")}</div>
      ${detail}
      ${payloadBlock}
    </article>`;
}

function tickTimestamps() {
  document.querySelectorAll(".flow-ts").forEach((el) => {
    const ts = el.getAttribute("data-ts");
    el.textContent = relativeTime(ts);
  });
}

// ---------------------------------------------------------------------------
// Wire up
// ---------------------------------------------------------------------------
function attachEventListeners() {
  $("#login-form").addEventListener("submit", handleLogin);
  $("#logout-btn").addEventListener("click", handleLogout);
  $("#chat-form").addEventListener("submit", handleChatSubmit);
  $("#flow-clear").addEventListener("click", () => {
    state.flowEvents = [];
    renderFlow();
  });
  $("#sim-arm-btn").addEventListener("click", handleSimArm);
  $("#stress-btn").addEventListener("click", handleStressFire);
  $("#new-chat-btn").addEventListener("click", handleNewChat);
  $("#reset-genie-btn").addEventListener("click", handleResetGenie);
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });
}

async function handleStressFire() {
  const count = parseInt($("#stress-count").value, 10) || 10;
  const btn = $("#stress-btn");
  const status = $("#stress-status");
  btn.disabled = true;
  const prev = btn.textContent;
  btn.textContent = `Firing ${count}...`;
  status.textContent = "";
  try {
    const resp = await api("/api/dev/stress-genie", {
      method: "POST",
      body: JSON.stringify({ count, question: "count sales" }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: `HTTP ${resp.status}` }));
      status.textContent = `failed: ${err.detail}`;
      return;
    }
    const data = await resp.json();
    status.textContent = `${data.ok} ok / ${data.failed} failed`;
  } catch (_) {
    status.textContent = "network error";
  } finally {
    btn.disabled = false;
    btn.textContent = prev;
  }
}

async function handleSimArm() {
  const count = parseInt($("#sim-count").value, 10) || 0;
  const btn = $("#sim-arm-btn");
  btn.disabled = true;
  const prev = btn.textContent;
  btn.textContent = "Arming...";
  try {
    const resp = await api("/api/dev/simulate-rate-limit", {
      method: "POST",
      body: JSON.stringify({ count, status: 429 }),
    });
    if (!resp.ok) {
      $("#sim-status").textContent = "arm failed";
      return;
    }
    const data = await resp.json();
    $("#sim-status").textContent =
      data.armed > 0
        ? `armed: next ${data.armed} will 429`
        : "simulation off";
  } catch (_) {
    $("#sim-status").textContent = "arm failed";
  } finally {
    btn.disabled = false;
    btn.textContent = prev;
  }
}

// ---------------------------------------------------------------------------
// Tabs (Chat | Architecture)
// ---------------------------------------------------------------------------
function switchTab(name) {
  if (name !== "main" && name !== "arch") return;
  state.tab = name;
  document.querySelectorAll(".tab").forEach((b) => {
    const active = b.dataset.tab === name;
    b.classList.toggle("active", active);
    b.setAttribute("aria-selected", active ? "true" : "false");
  });
  $("#tab-main").hidden = name !== "main";
  $("#tab-arch").hidden = name !== "arch";
  if (name === "arch") renderArchDiagrams();
}

async function renderArchDiagrams() {
  if (state.mermaidRendered) return;
  if (!window.mermaid) return; // script not yet loaded
  if (!state.mermaidInitialized) {
    window.mermaid.initialize({
      startOnLoad: false,
      theme: "default",
      securityLevel: "loose",
      flowchart: { htmlLabels: true, curve: "basis" },
      sequence: { mirrorActors: false, showSequenceNumbers: false },
    });
    state.mermaidInitialized = true;
  }
  try {
    await window.mermaid.run({ querySelector: "#tab-arch .mermaid" });
    state.mermaidRendered = true;
  } catch (e) {
    console.error("mermaid render failed", e);
    // If a diagram blew up, show the error in-place so it's not invisible.
    document.querySelectorAll("#tab-arch .mermaid").forEach((el) => {
      if (!el.querySelector("svg")) {
        el.innerHTML = `<div class="diagram-error">Diagram failed to render. Open browser console for details.</div>`;
      }
    });
  }
}

function init() {
  attachEventListeners();
  setInterval(tickTimestamps, 5000);
  tryRestoreSession();
}

init();
