"""Kairos Web UI — aiohttp-based web server with embedded SPA.

Single-page app with Linear-inspired dark theme.
Features: chat with SSE streaming, session management, tool visibility.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("kairos.web")


class WebServer:
    """Self-contained web server for the Kairos Web UI.

    Supports multiple concurrent sessions — each has its own StatefulAgent
    with isolated conversation history.

    Endpoints:
        GET  /           — Serve the SPA
        POST /api/chat   — Send message, returns SSE stream
        GET  /api/health — Health check
        GET  /api/sessions      — List saved sessions (persistent)
        POST /api/sessions/new  — Create a new session
        POST /api/sessions/load  — Load a persistent session
        POST /api/sessions/save  — Save current session
        DELETE /api/sessions/{name} — Delete a persistent session
        GET  /api/sessions/active — List active (in-memory) sessions
    """

    def __init__(
        self,
        agent: Any,
        host: str = "127.0.0.1",
        port: int = 8080,
        cors_origins: list[str] | None = None,
    ):
        self._agent_factory = agent  # template for creating new sessions
        self.host = host
        self.port = port
        self._cors_origins = cors_origins or ["*"]
        self._app = None
        self._runner = None
        self._started_at = time.time()
        # Multi-session support
        self._sessions: dict[str, Any] = {"default": agent}
        self._session_lock = asyncio.Lock()

    # ── Build app ──────────────────────────────────────────────

    def build_app(self):
        """Build the aiohttp application with all routes."""
        from aiohttp import web

        app = web.Application()
        app.router.add_get("/", self._handle_index)
        app.router.add_post("/api/chat", self._handle_chat)
        app.router.add_get("/api/health", self._handle_health)
        app.router.add_get("/api/sessions", self._handle_list_sessions)
        app.router.add_get("/api/sessions/active", self._handle_active_sessions)
        app.router.add_post("/api/sessions/new", self._handle_new_session)
        app.router.add_post("/api/sessions/load", self._handle_load_session)
        app.router.add_post("/api/sessions/save", self._handle_save_session)
        app.router.add_delete("/api/sessions/{name}", self._handle_delete_session)
        self._app = app
        return app

    # ── Serve SPA ──────────────────────────────────────────────

    async def _handle_index(self, request):
        from aiohttp import web
        return web.Response(
            text=SPA_HTML,
            content_type="text/html; charset=utf-8",
        )

    # ── Chat with SSE streaming ────────────────────────────────

    async def _handle_chat(self, request):
        from aiohttp import web

        try:
            body = await request.json()
            message = body.get("message", "").strip()
            session_id = body.get("session_id", "default")
            if not message:
                return web.json_response({"error": "Empty message"}, status=400)
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        # Resolve session
        agent = await self._get_session(session_id)

        # Prepare SSE response
        resp = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
            },
        )
        await resp.prepare(request)

        try:
            if hasattr(agent, "chat_stream"):
                stream = agent.chat_stream(message)
                for event in stream:
                    await resp.write(
                        f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode()
                    )
            else:
                result = agent.chat(message)
                event = {
                    "type": "done",
                    "content": result.get("content", str(result)),
                    "tool_calls": result.get("tool_calls", []),
                    "usage": result.get("usage", {}),
                }
                await resp.write(
                    f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode()
                )
        except Exception as e:
            logger.error("Chat stream error: %s", e)
            await resp.write(
                f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n".encode()
            )

        await resp.write_eof()
        return resp

    # ── Session management (multi-session) ─────────────────────

    async def _get_session(self, session_id: str) -> Any:
        """Get or create a session agent."""
        async with self._session_lock:
            if session_id not in self._sessions:
                # Create a fresh agent from the factory template
                from copy import deepcopy
                agent = deepcopy(self._agent_factory)
                # If it's a StatefulAgent, preserve its config
                if hasattr(agent, "reset"):
                    agent.reset()
                self._sessions[session_id] = agent
            return self._sessions[session_id]

    async def _handle_new_session(self, request):
        """Create a new session and return its ID."""
        from aiohttp import web
        import uuid
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        async with self._session_lock:
            from copy import deepcopy
            agent = deepcopy(self._agent_factory)
            if hasattr(agent, "reset"):
                agent.reset()
            self._sessions[session_id] = agent
        return web.json_response({"session_id": session_id, "status": "created"})

    async def _handle_active_sessions(self, request):
        """List all active (in-memory) sessions."""
        from aiohttp import web
        async with self._session_lock:
            sessions = list(self._sessions.keys())
        return web.json_response({"sessions": sessions, "count": len(sessions)})

    # ── Health ─────────────────────────────────────────────────

    async def _handle_health(self, request):
        from aiohttp import web
        async with self._session_lock:
            session_count = len(self._sessions)
        return web.json_response({
            "status": "ok",
            "uptime": time.time() - self._started_at,
            "active_sessions": session_count,
        })

    # ── Sessions ───────────────────────────────────────────────

    async def _handle_list_sessions(self, request):
        from aiohttp import web
        if hasattr(self.agent, "list_sessions"):
            sessions = self.agent.list_sessions()
        else:
            sessions = []
        return web.json_response({"sessions": sessions})

    async def _handle_load_session(self, request):
        from aiohttp import web
        try:
            body = await request.json()
            name = body.get("name", "")
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)
        if not name:
            return web.json_response({"error": "Session name required"}, status=400)
        if hasattr(self.agent, "load_session"):
            ok = self.agent.load_session(name)
            return web.json_response({"success": ok, "name": name})
        return web.json_response({"error": "Session loading not supported"}, status=501)

    async def _handle_save_session(self, request):
        from aiohttp import web
        try:
            body = await request.json()
            name = body.get("name", "current")
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)
        if hasattr(self.agent, "save_session"):
            self.agent.save_session(name)
            return web.json_response({"success": True, "name": name})
        return web.json_response({"error": "Session saving not supported"}, status=501)

    async def _handle_delete_session(self, request):
        from aiohttp import web
        name = request.match_info.get("name", "")
        if not name:
            return web.json_response({"error": "Session name required"}, status=400)
        if hasattr(self.agent, "delete_session"):
            ok = self.agent.delete_session(name)
            return web.json_response({"success": ok, "name": name})
        return web.json_response({"error": "Session deletion not supported"}, status=501)

    # ── Start / Stop ───────────────────────────────────────────

    async def start(self) -> None:
        """Start the web server."""
        from aiohttp import web

        if self._app is None:
            self.build_app()

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        logger.info("Kairos Web UI: http://%s:%d", self.host, self.port)

    async def stop(self) -> None:
        """Stop the web server."""
        if self._runner:
            await self._runner.cleanup()
            self._runner = None


# ═══════════════════════════════════════════════════════════════
# Embedded SPA — Linear-inspired dark theme
# ═══════════════════════════════════════════════════════════════

SPA_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Kairos — Web UI</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;510;590&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #08090a;
  --panel: #0f1011;
  --surface: #191a1b;
  --elevated: #23252a;
  --text: #f7f8f8;
  --text2: #d0d6e0;
  --text3: #8a8f98;
  --text4: #62666d;
  --accent: #5e6ad2;
  --accent-bright: #7170ff;
  --border: rgba(255,255,255,0.08);
  --border-subtle: rgba(255,255,255,0.05);
  --green: #27a644;
  --red: #e5484d;
  --yellow: #f5a623;
  --radius: 8px;
  --radius-sm: 6px;
  --font: 'Inter', system-ui, -apple-system, sans-serif;
  --mono: 'JetBrains Mono', ui-monospace, monospace;
  font-feature-settings: 'cv01', 'ss03';
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: var(--font);
  background: var(--bg);
  color: var(--text);
  height: 100vh;
  display: flex;
  overflow: hidden;
}

/* ── Sidebar ──────────────────────────── */
#sidebar {
  width: 260px;
  background: var(--panel);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
}
#sidebar-header {
  padding: 20px 16px 12px;
  border-bottom: 1px solid var(--border-subtle);
}
#sidebar-header h1 {
  font-size: 18px;
  font-weight: 590;
  letter-spacing: -0.24px;
  color: var(--text);
  display: flex;
  align-items: center;
  gap: 8px;
}
#sidebar-header .dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--green);
  display: inline-block;
}
#session-list {
  flex: 1;
  overflow-y: auto;
  padding: 8px;
}
.session-item {
  padding: 10px 12px;
  border-radius: var(--radius-sm);
  cursor: pointer;
  font-size: 13px;
  font-weight: 510;
  color: var(--text2);
  transition: background 0.15s;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.session-item:hover { background: var(--surface); }
.session-item.active { background: rgba(94,106,210,0.15); color: var(--accent-bright); }
.session-item .actions { display: none; gap: 4px; }
.session-item:hover .actions { display: flex; }
.session-item .actions button {
  background: none; border: none; color: var(--text4);
  cursor: pointer; font-size: 11px; padding: 2px 4px;
}
.session-item .actions button:hover { color: var(--red); }
#sidebar-footer {
  padding: 12px 16px;
  border-top: 1px solid var(--border-subtle);
  display: flex;
  gap: 8px;
}
#sidebar-footer button {
  flex: 1;
  padding: 8px 12px;
  border-radius: var(--radius-sm);
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.02);
  color: var(--text2);
  font-family: var(--font);
  font-size: 13px;
  font-weight: 510;
  cursor: pointer;
  transition: background 0.15s;
}
#sidebar-footer button:hover { background: var(--surface); }

/* ── Main Chat ────────────────────────── */
#main {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
}
#chat-header {
  padding: 16px 24px;
  border-bottom: 1px solid var(--border-subtle);
  display: flex;
  align-items: center;
  justify-content: space-between;
}
#chat-header .model-badge {
  font-size: 12px;
  font-weight: 510;
  color: var(--text3);
  padding: 4px 10px;
  border-radius: 9999px;
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.02);
}
#messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px 24px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.msg {
  max-width: 85%;
  animation: fadeIn 0.2s ease;
}
@keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }
.msg.user {
  align-self: flex-end;
  background: var(--surface);
  padding: 12px 16px;
  border-radius: var(--radius) var(--radius) 4px var(--radius);
  border: 1px solid var(--border);
}
.msg.user .role { display: none; }
.msg.agent {
  align-self: flex-start;
  padding: 12px 16px;
}
.msg.agent .role {
  font-size: 11px;
  font-weight: 590;
  color: var(--accent-bright);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 6px;
}
.msg .content {
  font-size: 15px;
  line-height: 1.6;
  color: var(--text2);
  white-space: pre-wrap;
  word-break: break-word;
}
.msg .content code {
  font-family: var(--mono);
  font-size: 13px;
  background: rgba(255,255,255,0.06);
  padding: 2px 6px;
  border-radius: 4px;
}
.msg .content pre {
  font-family: var(--mono);
  font-size: 13px;
  background: rgba(255,255,255,0.04);
  padding: 12px;
  border-radius: var(--radius-sm);
  border: 1px solid var(--border);
  overflow-x: auto;
  margin: 8px 0;
}

/* ── Tool calls ────────────────────────── */
.tool-calls {
  margin-top: 8px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.tool-call {
  font-size: 12px;
  padding: 6px 10px;
  border-radius: var(--radius-sm);
  background: rgba(255,255,255,0.03);
  border: 1px solid var(--border-subtle);
  cursor: pointer;
}
.tool-call .name { color: var(--yellow); font-weight: 510; }
.tool-call .result { color: var(--text4); margin-left: 6px; font-family: var(--mono); font-size: 11px; }
.tool-call .details { display: none; margin-top: 6px; }
.tool-call.expanded .details { display: block; }

/* ── Input area ────────────────────────── */
#input-area {
  padding: 16px 24px;
  border-top: 1px solid var(--border-subtle);
  display: flex;
  gap: 12px;
  align-items: flex-end;
}
#input-area textarea {
  flex: 1;
  background: rgba(255,255,255,0.02);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  color: var(--text);
  font-family: var(--font);
  font-size: 15px;
  padding: 12px 14px;
  resize: none;
  min-height: 44px;
  max-height: 200px;
  outline: none;
  line-height: 1.5;
  transition: border-color 0.15s;
}
#input-area textarea:focus { border-color: var(--accent-bright); }
#send-btn {
  width: 44px;
  height: 44px;
  border-radius: var(--radius-sm);
  border: none;
  background: var(--accent);
  color: white;
  cursor: pointer;
  font-size: 18px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background 0.15s;
  flex-shrink: 0;
}
#send-btn:hover { background: var(--accent-bright); }
#send-btn:disabled { opacity: 0.4; cursor: not-allowed; }

/* ── Toast ─────────────────────────────── */
#toast {
  position: fixed;
  bottom: 100px;
  right: 24px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 10px 16px;
  font-size: 13px;
  color: var(--text2);
  opacity: 0;
  transform: translateY(8px);
  transition: all 0.2s;
  pointer-events: none;
  z-index: 100;
}
#toast.show { opacity: 1; transform: translateY(0); }

/* ── Mobile ────────────────────────────── */
@media (max-width: 768px) {
  #sidebar { display: none; }
  #sidebar.open { display: flex; position: fixed; top: 0; left: 0; bottom: 0; z-index: 50; width: 280px; }
  .msg { max-width: 95%; }
  #messages { padding: 12px; }
  #input-area { padding: 12px; }
}
</style>
</head>
<body>

<!-- Sidebar -->
<aside id="sidebar">
  <div id="sidebar-header">
    <h1><span class="dot"></span> Kairos</h1>
  </div>
  <div id="session-list"></div>
  <div id="sidebar-footer">
    <button onclick="newSession()">+ New</button>
    <button onclick="saveSession()">💾 Save</button>
  </div>
</aside>

<!-- Main -->
<main id="main">
  <div id="chat-header">
    <span style="font-size:14px;font-weight:510;color:var(--text3)">Web UI</span>
    <span class="model-badge" id="model-badge">kairos</span>
    <span class="model-badge" id="session-id-display" style="font-size:11px;opacity:0.6">default</span>
  </div>
  <div id="messages">
    <div class="msg agent">
      <div class="role">Kairos</div>
      <div class="content">Welcome to Kairos Web UI. Type a message to start.</div>
    </div>
  </div>
  <div id="input-area">
    <textarea id="input" rows="1" placeholder="Type a message... (Enter to send, Shift+Enter for newline)"
              onkeydown="handleKey(event)"></textarea>
    <button id="send-btn" onclick="send()">→</button>
  </div>
</main>

<div id="toast"></div>

<script>
// ── State ────────────────────────────────
let currentSession = null;
let sessionId = 'default';  // backend session ID for multi-session
let streaming = false;

// ── Send message ─────────────────────────
async function send() {
  const input = document.getElementById('input');
  const msg = input.value.trim();
  if (!msg || streaming) return;
  input.value = '';
  input.style.height = 'auto';
  streaming = true;
  document.getElementById('send-btn').disabled = true;

  appendMessage('user', msg);
  const agentDiv = appendMessage('agent', '');

  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: msg, session_id: sessionId})
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let toolCalls = [];

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const event = JSON.parse(line.slice(6));
          if (event.type === 'token') {
            agentDiv.querySelector('.content').textContent += event.content;
            scrollDown();
          } else if (event.type === 'tool_call') {
            toolCalls.push(event);
          } else if (event.type === 'tool_result') {
            const tc = toolCalls.find(t => t.name === event.name);
            if (tc) tc.result = event.result;
            renderToolCalls(agentDiv, toolCalls);
          } else if (event.type === 'done') {
            if (event.tool_calls) {
              toolCalls = event.tool_calls;
              renderToolCalls(agentDiv, toolCalls);
            }
          } else if (event.type === 'error') {
            agentDiv.querySelector('.content').textContent += '\n\n⚠️ Error: ' + event.message;
          }
        } catch(e) {}
      }
    }
  } catch(e) {
    agentDiv.querySelector('.content').textContent += '\n\n⚠️ Connection error: ' + e.message;
  }

  streaming = false;
  document.getElementById('send-btn').disabled = false;
  refreshSessions();
}

// ── UI helpers ───────────────────────────
function appendMessage(role, content) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.innerHTML = '<div class="role">' + (role === 'user' ? 'You' : 'Kairos') + '</div>'
    + '<div class="content">' + escapeHtml(content) + '</div>';
  document.getElementById('messages').appendChild(div);
  scrollDown();
  return div;
}

function renderToolCalls(container, calls) {
  let existing = container.querySelector('.tool-calls');
  if (!existing) {
    existing = document.createElement('div');
    existing.className = 'tool-calls';
    container.appendChild(existing);
  }
  existing.innerHTML = calls.map((tc, i) =>
    '<div class="tool-call" onclick="this.classList.toggle(\'expanded\')">'
    + '<span class="name">🔧 ' + escapeHtml(tc.name || '?') + '</span>'
    + (tc.result ? '<span class="result">' + escapeHtml(String(tc.result).slice(0, 100)) + '</span>' : '')
    + '<div class="details"><pre>' + escapeHtml(JSON.stringify(tc.args || tc.arguments || {}, null, 2)) + '</pre></div>'
    + '</div>'
  ).join('');
}

function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function scrollDown() {
  const msgs = document.getElementById('messages');
  msgs.scrollTop = msgs.scrollHeight;
}

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    send();
  }
  // Auto-resize
  setTimeout(() => {
    const ta = e.target;
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 200) + 'px';
  }, 0);
}

// ── Sessions ─────────────────────────────
async function refreshSessions() {
  try {
    const resp = await fetch('/api/sessions');
    const data = await resp.json();
    const list = document.getElementById('session-list');
    list.innerHTML = data.sessions.map(s =>
      '<div class="session-item' + (currentSession === s.name ? ' active' : '') + '" onclick="loadSession(\'' + s.name + '\')">'
      + '<span>' + escapeHtml(s.name) + ' <span style="font-size:11px;color:var(--text4)">(' + (s.turn_count||0) + ' turns)</span></span>'
      + '<span class="actions"><button onclick="event.stopPropagation();deleteSession(\'' + s.name + '\')">✕</button></span>'
      + '</div>'
    ).join('');
  } catch(e) {}
}

async function loadSession(name) {
  try {
    await fetch('/api/sessions/load', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name})
    });
    currentSession = name;
    document.getElementById('messages').innerHTML = '';
    toast('Loaded: ' + name);
    refreshSessions();
  } catch(e) {}
}

async function saveSession() {
  const name = prompt('Session name:', currentSession || 'session-' + Date.now().toString(36));
  if (!name) return;
  try {
    await fetch('/api/sessions/save', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name})
    });
    currentSession = name;
    toast('Saved: ' + name);
    refreshSessions();
  } catch(e) {}
}

async function deleteSession(name) {
  if (!confirm('Delete session "' + name + '"?')) return;
  try {
    await fetch('/api/sessions/' + encodeURIComponent(name), {method: 'DELETE'});
    if (currentSession === name) currentSession = null;
    toast('Deleted: ' + name);
    refreshSessions();
  } catch(e) {}
}

async function newSession() {
  try {
    const resp = await fetch('/api/sessions/new', {method: 'POST'});
    const data = await resp.json();
    sessionId = data.session_id;
    document.getElementById('messages').innerHTML = '';
    currentSession = null;
    document.getElementById('session-id-display').textContent = sessionId;
    toast('New session: ' + sessionId);
  } catch(e) {
    toast('Failed to create session');
  }
}

function toast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2000);
}

// ── Init ─────────────────────────────────
refreshSessions();
</script>
</body>
</html>"""
