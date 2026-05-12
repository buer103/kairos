"""Kairos Web UI — aiohttp-based web server with embedded SPA.

Production-grade single-page app with Linear-inspired dark theme.
Features: multi-session, live streaming, rich tool cards, markdown rendering,
          code highlighting, mobile responsive, keyboard shortcuts.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

logger = logging.getLogger("kairos.web")


class WebServer:
    """Self-contained web server for the Kairos Web UI.

    Supports multiple concurrent sessions — each has its own StatefulAgent
    with isolated conversation history.

    Endpoints:
        GET  /                    — Serve the SPA
        POST /api/chat            — Send message, returns SSE stream
        GET  /api/health          — Health check
        GET  /api/sessions        — List saved sessions
        POST /api/sessions/new    — Create new session
        POST /api/sessions/load   — Load session
        POST /api/sessions/save   — Save session
        DELETE /api/sessions/{name} — Delete session
        GET  /api/sessions/active  — List active sessions
    """

    def __init__(
        self,
        agent: Any,
        host: str = "127.0.0.1",
        port: int = 8080,
        cors_origins: list[str] | None = None,
    ):
        self._agent_factory = agent
        self.host = host
        self.port = port
        self._cors_origins = cors_origins or ["*"]
        self._app = None
        self._runner = None
        self._started_at = time.time()
        self._sessions: dict[str, Any] = {"default": agent}
        self._session_lock = asyncio.Lock()

    # ── Build app ──────────────────────────────────────────────

    def build_app(self):
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
        return web.Response(text=SPA_HTML, content_type="text/html; charset=utf-8")

    # ── Chat (SSE streaming) ───────────────────────────────────

    async def _handle_chat(self, request):
        from aiohttp import web

        try:
            body = await request.json()
            message = body.get("message", "")
            session_id = body.get("session_id", "default")
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        if not message:
            return web.json_response({"error": "Message required"}, status=400)

        # Get or create session agent
        async with self._session_lock:
            agent = self._sessions.get(session_id)
            if agent is None:
                agent = self._create_session_agent()
                self._sessions[session_id] = agent

        response = web.StreamResponse()
        response.headers["Content-Type"] = "text/event-stream"
        response.headers["Cache-Control"] = "no-cache"
        response.headers["X-Accel-Buffering"] = "no"
        await response.prepare(request)

        try:
            if hasattr(agent, "chat_stream"):
                for event in agent.chat_stream(message):
                    data = json.dumps(event, default=str)
                    await response.write(f"data: {data}\n\n".encode())
            else:
                result = agent.run(message)
                await response.write(
                    f"data: {json.dumps({'type': 'done', 'content': result.get('content', ''), 'tool_calls': result.get('tool_calls', [])}, default=str)}\n\n".encode()
                )
        except Exception as e:
            logger.exception("Chat error")
            await response.write(
                f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n".encode()
            )

        await response.write_eof()
        return response

    def _create_session_agent(self):
        """Create a fresh agent for a new session."""
        import copy
        try:
            return copy.deepcopy(self._agent_factory)
        except Exception:
            return self._agent_factory

    # ── Health ─────────────────────────────────────────────────

    async def _handle_health(self, request):
        from aiohttp import web
        return web.json_response({
            "status": "ok",
            "uptime": time.time() - self._started_at,
            "sessions": len(self._sessions),
        })

    # ── Session management ─────────────────────────────────────

    async def _handle_list_sessions(self, request):
        from aiohttp import web
        agent = self._sessions.get("default", self._agent_factory)
        if hasattr(agent, "list_sessions"):
            sessions = agent.list_sessions()
            return web.json_response({"sessions": sessions})
        return web.json_response({"sessions": []})

    async def _handle_active_sessions(self, request):
        from aiohttp import web
        async with self._session_lock:
            active = list(self._sessions.keys())
        return web.json_response({"active": active, "count": len(active)})

    async def _handle_new_session(self, request):
        from aiohttp import web
        import uuid

        session_id = f"session-{uuid.uuid4().hex[:8]}"
        async with self._session_lock:
            agent = self._create_session_agent()
            self._sessions[session_id] = agent
        return web.json_response({"session_id": session_id, "active": len(self._sessions)})

    async def _handle_load_session(self, request):
        from aiohttp import web
        try:
            body = await request.json()
            name = body.get("name", "")
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)
        if not name:
            return web.json_response({"error": "Session name required"}, status=400)
        if hasattr(self._agent_factory, "load_session"):
            ok = self._agent_factory.load_session(name)
            return web.json_response({"success": ok, "name": name})
        return web.json_response({"error": "Session loading not supported"}, status=501)

    async def _handle_save_session(self, request):
        from aiohttp import web
        try:
            body = await request.json()
            name = body.get("name", "current")
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)
        if hasattr(self._agent_factory, "save_session"):
            self._agent_factory.save_session(name)
            return web.json_response({"success": True, "name": name})
        return web.json_response({"error": "Session saving not supported"}, status=501)

    async def _handle_delete_session(self, request):
        from aiohttp import web
        name = request.match_info.get("name", "")
        if not name:
            return web.json_response({"error": "Session name required"}, status=400)
        if hasattr(self._agent_factory, "delete_session"):
            ok = self._agent_factory.delete_session(name)
            return web.json_response({"success": ok, "name": name})
        return web.json_response({"error": "Session deletion not supported"}, status=501)

    # ── Start / Stop ───────────────────────────────────────────

    async def start(self) -> None:
        from aiohttp import web

        if self._app is None:
            self.build_app()

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        logger.info("Kairos Web UI: http://%s:%d", self.host, self.port)

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()
            self._runner = None


# ═══════════════════════════════════════════════════════════════
# Embedded SPA — Linear-inspired dark theme, production UX
# ═══════════════════════════════════════════════════════════════

SPA_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Kairos — Web UI</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;510;590;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
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
  --accent-dim: rgba(94,106,210,0.15);
  --border: rgba(255,255,255,0.08);
  --border-subtle: rgba(255,255,255,0.05);
  --green: #27a644;
  --green-bg: rgba(39,166,68,0.12);
  --red: #e5484d;
  --red-bg: rgba(229,72,77,0.12);
  --yellow: #f5a623;
  --yellow-bg: rgba(245,166,35,0.12);
  --blue: #4d7cff;
  --radius: 8px;
  --radius-sm: 6px;
  --font: 'Inter', system-ui, -apple-system, sans-serif;
  --mono: 'JetBrains Mono', ui-monospace, monospace;
  font-feature-settings: 'cv01','ss03';
}
* { margin:0; padding:0; box-sizing:border-box; }
body {
  font-family:var(--font);
  background:var(--bg);
  color:var(--text);
  height:100vh;
  display:flex;
  overflow:hidden;
}

/* ── Sidebar ──────────────────────────── */
#sidebar {
  width:272px; background:var(--panel); border-right:1px solid var(--border);
  display:flex; flex-direction:column; flex-shrink:0; transition:transform .2s;
}
#sidebar-header {
  padding:20px 16px 14px; border-bottom:1px solid var(--border-subtle);
}
#sidebar-header h1 {
  font-size:18px; font-weight:590; letter-spacing:-.24px; color:var(--text);
  display:flex; align-items:center; gap:8px;
}
#sidebar-header .dot {
  width:8px; height:8px; border-radius:50%; background:var(--green); display:inline-block;
}
#sidebar-header .subtitle {
  font-size:11px; color:var(--text4); font-weight:400; margin-top:4px; letter-spacing:0;
}
#session-list { flex:1; overflow-y:auto; padding:8px; }
.session-item {
  padding:10px 12px; border-radius:var(--radius-sm); cursor:pointer;
  font-size:13px; font-weight:510; color:var(--text2);
  transition:background .15s; display:flex; justify-content:space-between; align-items:center;
  margin-bottom:2px;
}
.session-item:hover { background:var(--surface); }
.session-item.active { background:var(--accent-dim); color:var(--accent-bright); }
.session-item .name { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.session-item .turns { font-size:11px; color:var(--text4); font-weight:400; margin-left:8px; }
.session-item .actions { display:none; gap:4px; }
.session-item:hover .actions { display:flex; }
.session-item .actions button {
  background:none; border:none; color:var(--text4);
  cursor:pointer; font-size:14px; padding:2px 6px; border-radius:4px;
}
.session-item .actions button:hover { color:var(--red); background:var(--red-bg); }
#sidebar-footer {
  padding:12px 16px; border-top:1px solid var(--border-subtle);
  display:flex; gap:8px;
}
#sidebar-footer button {
  flex:1; padding:8px 12px; border-radius:var(--radius-sm);
  border:1px solid var(--border); background:rgba(255,255,255,.02);
  color:var(--text2); font-family:var(--font); font-size:13px; font-weight:510;
  cursor:pointer; transition:background .15s; white-space:nowrap;
}
#sidebar-footer button:hover { background:var(--surface); }
#sidebar-footer button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
#sidebar-footer button.primary:hover { background:var(--accent-bright); }

/* ── Main ──────────────────────────────── */
#main { flex:1; display:flex; flex-direction:column; min-width:0; }
#chat-header {
  padding:14px 24px; border-bottom:1px solid var(--border-subtle);
  display:flex; align-items:center; gap:12px;
}
#chat-header .title { font-size:14px; font-weight:590; color:var(--text2); letter-spacing:-.2px; }
#chat-header .badge {
  font-size:11px; font-weight:510; color:var(--text3);
  padding:3px 10px; border-radius:9999px; border:1px solid var(--border);
  background:rgba(255,255,255,.02);
}
#chat-header .spacer { flex:1; }
#menu-toggle {
  display:none; background:none; border:none; color:var(--text3);
  font-size:20px; cursor:pointer; padding:4px 8px; border-radius:4px;
}
#menu-toggle:hover { background:var(--surface); }

/* ── Messages ──────────────────────────── */
#messages {
  flex:1; overflow-y:auto; padding:24px; display:flex; flex-direction:column; gap:20px;
  scroll-behavior:smooth;
}
#messages::-webkit-scrollbar { width:6px; }
#messages::-webkit-scrollbar-track { background:transparent; }
#messages::-webkit-scrollbar-thumb { background:var(--border); border-radius:3px; }

/* ── Welcome panel ─────────────────────── */
.welcome {
  align-self:center; text-align:center; padding:60px 20px 40px; max-width:520px;
  animation:fadeIn .4s ease;
}
.welcome .logo {
  font-size:40px; font-weight:700; letter-spacing:-1px;
  background:linear-gradient(135deg, var(--accent-bright), #a78bfa);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  margin-bottom:12px;
}
.welcome .tagline { font-size:16px; color:var(--text3); margin-bottom:32px; font-weight:400; }
.welcome .caps {
  display:grid; grid-template-columns:1fr 1fr; gap:8px; text-align:left;
}
.welcome .cap {
  display:flex; align-items:flex-start; gap:8px; padding:10px 12px;
  border-radius:var(--radius-sm); background:rgba(255,255,255,.015);
  border:1px solid var(--border-subtle); font-size:13px; color:var(--text3);
}
.welcome .cap .icon { font-size:16px; flex-shrink:0; margin-top:1px; }
.welcome .cap .label { font-weight:510; color:var(--text2); display:block; margin-bottom:2px; }
.welcome .hint {
  margin-top:24px; font-size:12px; color:var(--text4);
  border-top:1px solid var(--border-subtle); padding-top:16px;
}
.welcome .hint kbd {
  font-family:var(--mono); font-size:11px; padding:1px 6px; border-radius:4px;
  border:1px solid var(--border); background:rgba(255,255,255,.03); color:var(--text3);
}

/* ── Chat messages ─────────────────────── */
.msg {
  max-width:82%; animation:fadeIn .2s ease;
}
@keyframes fadeIn {
  from { opacity:0; transform:translateY(4px); }
  to { opacity:1; transform:translateY(0); }
}
.msg.user {
  align-self:flex-end; background:var(--surface); padding:14px 18px;
  border-radius:var(--radius) var(--radius) 4px var(--radius);
  border:1px solid var(--border);
}
.msg.user .role { display:none; }
.msg.agent {
  align-self:flex-start; padding:14px 18px;
}
.msg.agent .role {
  font-size:11px; font-weight:590; color:var(--accent-bright);
  text-transform:uppercase; letter-spacing:.5px; margin-bottom:8px;
  display:flex; align-items:center; gap:8px;
}
.msg.agent .role .copy-btn {
  font-size:11px; font-weight:400; background:none; border:1px solid var(--border);
  color:var(--text4); cursor:pointer; padding:2px 8px; border-radius:4px;
  text-transform:none; letter-spacing:0;
}
.msg.agent .role .copy-btn:hover { color:var(--text2); border-color:var(--text4); }
.msg .content {
  font-size:15px; line-height:1.65; color:var(--text2); white-space:pre-wrap; word-break:break-word;
}
.msg .content p { margin:6px 0; }
.msg .content p:first-child { margin-top:0; }
.msg .content code {
  font-family:var(--mono); font-size:13px; background:rgba(255,255,255,.06);
  padding:2px 6px; border-radius:4px;
}
.msg .content pre {
  font-family:var(--mono); font-size:13px; background:rgba(255,255,255,.04);
  padding:14px; border-radius:var(--radius-sm); border:1px solid var(--border);
  overflow-x:auto; margin:10px 0; line-height:1.5;
}
.msg .content pre code {
  background:none; padding:0; font-size:13px;
}
.msg .content ul, .msg .content ol { padding-left:20px; margin:6px 0; }
.msg .content a { color:var(--accent-bright); }
.msg .content blockquote {
  border-left:2px solid var(--accent); padding:6px 0 6px 14px;
  margin:8px 0; color:var(--text3);
}

/* ── Streaming cursor ──────────────────── */
.msg.streaming .content::after {
  content:''; display:inline-block; width:8px; height:16px; background:var(--accent-bright);
  vertical-align:text-bottom; margin-left:2px; animation:blink .8s infinite; border-radius:1px;
}
@keyframes blink { 0%,100% { opacity:1; } 50% { opacity:0; } }

/* ── Tool calls ────────────────────────── */
.tool-calls { margin-top:10px; display:flex; flex-direction:column; gap:6px; }
.tool-call {
  border-radius:var(--radius-sm); border:1px solid var(--border-subtle);
  background:rgba(255,255,255,.015); overflow:hidden; transition:border-color .15s;
}
.tool-call:hover { border-color:var(--border); }
.tool-call-header {
  padding:8px 12px; display:flex; align-items:center; gap:8px;
  cursor:pointer; font-size:12px; user-select:none;
}
.tool-call-header .icon { font-size:14px; flex-shrink:0; }
.tool-call-header .name { color:var(--yellow); font-weight:510; font-family:var(--mono); font-size:11px; }
.tool-call-header .status {
  font-size:11px; padding:2px 8px; border-radius:9999px; font-weight:510; flex-shrink:0;
}
.tool-call-header .status.success { background:var(--green-bg); color:var(--green); }
.tool-call-header .status.error { background:var(--red-bg); color:var(--red); }
.tool-call-header .status.running { background:var(--yellow-bg); color:var(--yellow); }
.tool-call-header .time { font-size:11px; color:var(--text4); margin-left:auto; }
.tool-call-header .expand-icon { font-size:10px; color:var(--text4); margin-left:4px; transition:transform .15s; }
.tool-call.expanded .expand-icon { transform:rotate(180deg); }
.tool-call-body {
  display:none; padding:0 12px 10px; font-family:var(--mono); font-size:11px;
  color:var(--text3); max-height:300px; overflow:auto;
}
.tool-call.expanded .tool-call-body { display:block; }
.tool-call-body pre {
  margin:0; white-space:pre-wrap; word-break:break-all;
  background:rgba(0,0,0,.2); padding:10px; border-radius:4px;
}

/* ── Input ─────────────────────────────── */
#input-area {
  padding:16px 24px; border-top:1px solid var(--border-subtle);
  display:flex; gap:12px; align-items:flex-end; background:var(--panel);
}
#input-area textarea {
  flex:1; background:var(--surface); border:1px solid var(--border);
  border-radius:var(--radius-sm); color:var(--text); font-family:var(--font);
  font-size:15px; padding:12px 14px; resize:none; min-height:44px; max-height:200px;
  outline:none; line-height:1.5; transition:border-color .15s;
}
#input-area textarea:focus { border-color:var(--accent-bright); }
#input-area textarea::placeholder { color:var(--text4); }
#send-btn {
  width:44px; height:44px; border-radius:var(--radius-sm); border:none;
  background:var(--accent); color:white; cursor:pointer; font-size:16px;
  display:flex; align-items:center; justify-content:center;
  transition:background .15s,opacity .15s; flex-shrink:0; font-weight:590;
}
#send-btn:hover { background:var(--accent-bright); }
#send-btn:disabled { opacity:.35; cursor:not-allowed; }
#send-btn .spinner {
  display:none; width:18px; height:18px; border:2px solid rgba(255,255,255,.2);
  border-top-color:#fff; border-radius:50%; animation:spin .6s linear infinite;
}
#send-btn.sending .arrow { display:none; }
#send-btn.sending .spinner { display:block; }
@keyframes spin { to { transform:rotate(360deg); } }

/* ── Toast ─────────────────────────────── */
#toast {
  position:fixed; bottom:100px; right:24px;
  background:var(--elevated); border:1px solid var(--border);
  border-radius:var(--radius-sm); padding:10px 16px;
  font-size:13px; color:var(--text2); font-weight:510;
  opacity:0; transform:translateY(8px); transition:all .25s;
  pointer-events:none; z-index:100; max-width:320px;
  box-shadow:0 8px 32px rgba(0,0,0,.4);
}
#toast.show { opacity:1; transform:translateY(0); }
#toast.success { border-color:rgba(39,166,68,.3); }
#toast.error { border-color:rgba(229,72,77,.3); }

/* ── Mobile ────────────────────────────── */
@media (max-width:768px) {
  #sidebar {
    position:fixed; top:0; left:0; bottom:0; z-index:50; width:280px;
    transform:translateX(-100%);
  }
  #sidebar.open { transform:translateX(0); box-shadow:4px 0 20px rgba(0,0,0,.5); }
  #overlay {
    display:none; position:fixed; inset:0; background:rgba(0,0,0,.4); z-index:49;
  }
  #overlay.show { display:block; }
  #menu-toggle { display:flex; }
  .msg { max-width:92%; }
  #messages { padding:16px; }
  #input-area { padding:12px; }
  .welcome { padding:40px 16px 30px; }
  .welcome .caps { grid-template-columns:1fr; }
}

/* ── Empty state ───────────────────────── */
.empty-state {
  align-self:center; text-align:center; padding:40px;
  color:var(--text4); font-size:14px;
}
</style>
</head>
<body>

<!-- Overlay for mobile -->
<div id="overlay" onclick="closeSidebar()"></div>

<!-- Sidebar -->
<aside id="sidebar">
  <div id="sidebar-header">
    <h1><span class="dot"></span> Kairos</h1>
    <div class="subtitle">Multi-session agent</div>
  </div>
  <div id="session-list">
    <div class="session-item active" onclick="switchToSession('default')">
      <span class="name">Default</span><span class="turns">active</span>
    </div>
  </div>
  <div id="sidebar-footer">
    <button class="primary" onclick="newSession()">+ New</button>
    <button onclick="saveSession()">Save</button>
  </div>
</aside>

<!-- Main -->
<main id="main">
  <div id="chat-header">
    <button id="menu-toggle" onclick="toggleSidebar()" title="Menu">☰</button>
    <span class="title">Chat</span>
    <span class="spacer"></span>
    <span class="badge" id="session-badge">default</span>
  </div>
  <div id="messages"><div class="welcome" id="welcome-screen"></div></div>
  <div id="input-area">
    <textarea id="input" rows="1" placeholder="Ask Kairos anything... (Enter to send)"
              onkeydown="handleKey(event)"></textarea>
    <button id="send-btn" onclick="send()"><span class="arrow">→</span><div class="spinner"></div></button>
  </div>
</main>

<div id="toast"></div>

<script>
// ── State ────────────────────────────────
let currentSession = null;
let sessionId = 'default';
let streaming = false;

// ── Welcome ──────────────────────────────
function showWelcome() {
  // Fetch session info
  fetch('/api/health').then(r=>r.json()).then(d => {
    document.getElementById('welcome-screen').innerHTML = 
      '<div class="logo">Kairos</div>' +
      '<div class="tagline">The right tool, at the right moment</div>' +
      '<div class="caps">' +
        '<div class="cap"><span class="icon">💬</span><div><span class="label">Multi-turn Chat</span>Stateful conversations with session persistence</div></div>' +
        '<div class="cap"><span class="icon">🔧</span><div><span class="label">23 Built-in Tools</span>File ops, terminal, web, browser, MCP, vision</div></div>' +
        '<div class="cap"><span class="icon">🧠</span><div><span class="label">20-Layer Middleware</span>Context compression, evidence tracking, confidence</div></div>' +
        '<div class="cap"><span class="icon">🔒</span><div><span class="label">Permission Control</span>3-level interactive permissions with /yolo</div></div>' +
      '</div>' +
      '<div class="hint">Active sessions: <b>' + d.sessions + '</b> · Press <kbd>Enter</kbd> to send · <kbd>Shift+Enter</kbd> for newline</div>';
  }).catch(()=>{});
}
showWelcome();

// ── Send ─────────────────────────────────
async function send() {
  const input = document.getElementById('input');
  const msg = input.value.trim();
  if (!msg || streaming) return;
  input.value = '';
  input.style.height = 'auto';
  streaming = true;

  const btn = document.getElementById('send-btn');
  btn.classList.add('sending');

  // Hide welcome
  const w = document.getElementById('welcome-screen');
  if (w) w.style.display = 'none';

  appendMessage('user', msg);
  const agentDiv = appendMessage('agent', '');
  agentDiv.classList.add('streaming');

  let fullContent = '';
  const toolCalls = [];
  let toolCallIndex = 0;

  try {
    const resp = await fetch('/api/chat', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message:msg, session_id:sessionId})
    });
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream:true});
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const event = JSON.parse(line.slice(6));
          switch (event.type) {
            case 'token':
              fullContent += event.content;
              agentDiv.querySelector('.content').innerHTML = renderMarkdown(fullContent);
              scrollDown();
              break;
            case 'tool_call':
              toolCalls.push({...event, id:++toolCallIndex, status:'running', startTime:Date.now()});
              renderToolCalls(agentDiv, toolCalls);
              break;
            case 'tool_result':
              const tc = toolCalls.find(t => t.name === event.name && t.status === 'running');
              if (tc) {
                tc.status = event.error ? 'error' : 'success';
                tc.result = event.result || event.error;
                tc.duration = Date.now() - (tc.startTime || Date.now());
                renderToolCalls(agentDiv, toolCalls);
              }
              break;
            case 'done':
              if (event.tool_calls) {
                for (const dtc of event.tool_calls) {
                  const existing = toolCalls.find(t => t.name === dtc.name);
                  if (existing) { existing.status = 'success'; existing.args = dtc.args || dtc.arguments; }
                  else toolCalls.push({...dtc, id:++toolCallIndex, status:'success'});
                }
                renderToolCalls(agentDiv, toolCalls);
              }
              if (event.content) {
                fullContent = event.content;
                agentDiv.querySelector('.content').innerHTML = renderMarkdown(fullContent);
              }
              break;
            case 'error':
              agentDiv.querySelector('.content').innerHTML += 
                '<div style="margin-top:12px;padding:10px;border-radius:6px;background:var(--red-bg);color:var(--red);font-size:13px">' +
                '⚠️ ' + escapeHtml(event.message) + '</div>';
              break;
          }
        } catch(e) {}
      }
    }
  } catch(e) {
    agentDiv.querySelector('.content').innerHTML += 
      '<div style="margin-top:12px;padding:10px;border-radius:6px;background:var(--red-bg);color:var(--red);font-size:13px">' +
      '⚠️ Connection error</div>';
  }

  agentDiv.classList.remove('streaming');
  streaming = false;
  btn.classList.remove('sending');
  refreshSessions();
}

// ── Markdown ─────────────────────────────
function renderMarkdown(text) {
  if (!text) return '';
  try {
    if (typeof marked !== 'undefined') {
      marked.setOptions({breaks:true, gfm:true});
      return marked.parse(text);
    }
  } catch(e) {}
  return escapeHtml(text);
}

// ── UI Helpers ───────────────────────────
function appendMessage(role, content) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.innerHTML = '<div class="role">' + (role==='user' ? 'You' : 
    'Kairos <button class="copy-btn" onclick="copyMessage(this)">📋 Copy</button>') + 
    '</div><div class="content">' + (role==='user' ? escapeHtml(content) : renderMarkdown(content)) + '</div>';
  document.getElementById('messages').appendChild(div);
  scrollDown();
  return div;
}

function copyMessage(btn) {
  const content = btn.closest('.msg').querySelector('.content').textContent;
  navigator.clipboard.writeText(content).then(() => {
    btn.textContent = '✓ Copied';
    setTimeout(() => { btn.textContent = '📋 Copy'; }, 2000);
  });
}

function renderToolCalls(container, calls) {
  let existing = container.querySelector('.tool-calls');
  if (!existing) {
    existing = document.createElement('div');
    existing.className = 'tool-calls';
    container.appendChild(existing);
  }
  existing.innerHTML = calls.map(tc => {
    const iconMap = {
      read_file:'📖', search_files:'🔍', write_file:'✏️', patch:'🔧', terminal:'💻',
      web_search:'🌐', web_fetch:'📥', delegate_task:'🤖', skill_view:'📚',
      browser:'🖥️', vision_analyze:'👁️', memory:'🧠', session_search:'📅'
    };
    const icon = iconMap[tc.name] || '🔧';
    const statusClass = tc.status || 'running';
    const statusLabel = {running:'Running', success:'Done', error:'Failed'}[statusClass] || statusClass;
    const timeStr = tc.duration ? (tc.duration < 1000 ? tc.duration+'ms' : (tc.duration/1000).toFixed(1)+'s') : '';
    return '<div class="tool-call" onclick="this.classList.toggle(\'expanded\')">' +
      '<div class="tool-call-header">' +
        '<span class="icon">' + icon + '</span>' +
        '<span class="name">' + escapeHtml(tc.name) + '</span>' +
        '<span class="status ' + statusClass + '">' + statusLabel + '</span>' +
        (timeStr ? '<span class="time">' + timeStr + '</span>' : '') +
        '<span class="expand-icon">▼</span>' +
      '</div>' +
      '<div class="tool-call-body"><pre>' + 
        escapeHtml(JSON.stringify(tc.args || tc.arguments || {}, null, 2)) + 
        (tc.result ? '\n\n→ Result: ' + escapeHtml(String(tc.result).slice(0,500)) : '') +
      '</pre></div>' +
    '</div>';
  }).join('');
}

function escapeHtml(s) {
  if (!s) return '';
  const d = document.createElement('div');
  d.textContent = String(s);
  return d.innerHTML;
}

function scrollDown() {
  const msgs = document.getElementById('messages');
  requestAnimationFrame(() => { msgs.scrollTop = msgs.scrollHeight; });
}

// ── Input ────────────────────────────────
function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    send();
  }
  setTimeout(() => {
    const ta = e.target;
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 200) + 'px';
  }, 0);
}

// ── Sidebar ──────────────────────────────
function toggleSidebar() {
  document.getElementById('sidebar').classList.toggle('open');
  document.getElementById('overlay').classList.toggle('show');
}
function closeSidebar() {
  document.getElementById('sidebar').classList.remove('open');
  document.getElementById('overlay').classList.remove('show');
}

// ── Sessions ─────────────────────────────
async function refreshSessions() {
  try {
    const [savedResp, activeResp] = await Promise.all([
      fetch('/api/sessions'),
      fetch('/api/sessions/active')
    ]);
    const saved = await savedResp.json();
    const active = await activeResp.json();

    const list = document.getElementById('session-list');
    const sessions = new Map();

    // Active sessions
    for (const id of active.active || []) {
      sessions.set(id, {name:id, turns:0, active:true});
    }
    // Saved sessions
    for (const s of saved.sessions || []) {
      if (!sessions.has(s.name)) sessions.set(s.name, {...s, active:false});
      else { const e = sessions.get(s.name); e.turns = s.turn_count || 0; e.active = true; }
    }

    // Always include current
    if (!sessions.has(sessionId)) sessions.set(sessionId, {name:sessionId, turns:0, active:true});

    list.innerHTML = Array.from(sessions.values()).map(s =>
      '<div class="session-item' + (sessionId === s.name ? ' active' : '') + 
      '" onclick="switchToSession(\'' + escapeHtml(s.name) + '\')">' +
      '<span class="name">' + escapeHtml(s.name) +
        (s.active ? ' <span style="font-size:9px;color:var(--green)">●</span>' : '') +
      '</span>' +
      '<span class="turns">' + (s.turns ? s.turns + ' turns' : '') + '</span>' +
      '<span class="actions">' +
        '<button onclick="event.stopPropagation();deleteSession(\'' + escapeHtml(s.name) + '\')" title="Delete">✕</button>' +
      '</span>' +
      '</div>'
    ).join('');

    document.getElementById('session-badge').textContent = sessionId;
  } catch(e) {}
}

async function switchToSession(name) {
  if (name === sessionId) return;
  try {
    if (name !== 'default') {
      await fetch('/api/sessions/load', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body:JSON.stringify({name})
      });
    }
    sessionId = name;
    document.getElementById('messages').innerHTML = '';
    showWelcome();
    toast('Switched to: ' + name, 'success');
    refreshSessions();
    closeSidebar();
  } catch(e) { toast('Failed to switch session', 'error'); }
}

async function saveSession() {
  const name = prompt('Session name:', currentSession || 'session-' + Date.now().toString(36));
  if (!name) return;
  try {
    await fetch('/api/sessions/save', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({name})
    });
    currentSession = name;
    toast('Saved: ' + name, 'success');
    refreshSessions();
  } catch(e) { toast('Failed to save session', 'error'); }
}

async function deleteSession(name) {
  if (!confirm('Delete session "' + name + '"? This cannot be undone.')) return;
  try {
    await fetch('/api/sessions/' + encodeURIComponent(name), {method:'DELETE'});
    if (currentSession === name) currentSession = null;
    toast('Deleted: ' + name);
    refreshSessions();
  } catch(e) { toast('Failed to delete session', 'error'); }
}

async function newSession() {
  try {
    const resp = await fetch('/api/sessions/new', {method:'POST'});
    const data = await resp.json();
    sessionId = data.session_id;
    document.getElementById('messages').innerHTML = '';
    showWelcome();
    currentSession = null;
    toast('New session created', 'success');
    refreshSessions();
    closeSidebar();
  } catch(e) { toast('Failed to create session', 'error'); }
}

// ── Toast ────────────────────────────────
let _toastTimer;
function toast(msg, type) {
  const t = document.getElementById('toast');
  clearTimeout(_toastTimer);
  t.textContent = msg;
  t.className = 'show' + (type ? ' ' + type : '');
  _toastTimer = setTimeout(() => t.classList.remove('show'), 2500);
}

// ── Init ─────────────────────────────────
refreshSessions();
</script>
</body>
</html>"""