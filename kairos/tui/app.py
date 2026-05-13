"""Textual TUI app for Kairos — multi-panel agent chat interface.

Phase 4: skin hot-switching, keyboard shortcuts, YOLO mode

Layout:
    ┌──────────┬──────────────────────────────────────┐
    │ Sidebar  │  Header (thinking… model)            │
    │ Sessions │  ─────────────────────────────────── │
    │ Tools    │  Transcript (scrollable)             │
    │ Model    │                                      │
    │          ├──────────────────────────────────────│
    │          │  Status (model tokens cost)          │
    │          ├──────────────────────────────────────│
    │          │  Input bar                           │
    └──────────┴──────────────────────────────────────┘

Key bindings:
    Ctrl+C      — Interrupt agent
    Ctrl+L      — Clear transcript
    Ctrl+S      — Save session
    Ctrl+Q      — Quit
    Escape      — Clear input / cancel

Slash commands: /help /exit /clear /skin /model /tools /sessions /yolo /verbose
"""

from __future__ import annotations

import time
from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Static

from kairos.tui.widgets.transcript import Transcript
from kairos.tui.widgets.input_bar import InputBar
from kairos.tui.widgets.sidebar import Sidebar
from kairos.tui.theme import get_skin, build_skin_css


class KairosTUI(App):
    """Main Textual App — multi-panel agent chat with skins."""

    CSS_PATH = "styles/app.tcss"
    TITLE = "Kairos TUI"
    SUB_TITLE = "The right tool, at the right moment"

    BINDINGS = [
        Binding("ctrl+c", "interrupt", "Interrupt", show=True),
        Binding("ctrl+l", "clear_screen", "Clear", show=True),
        Binding("ctrl+s", "save_session", "Save", show=True),
        Binding("ctrl+q", "quit", "Quit", show=True),
        Binding("escape", "cancel_input", "Cancel", show=False),
    ]

    def __init__(
        self,
        agent: Any = None,
        skin: str = "default",
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self._agent = agent
        self._skin_name = skin
        self._skin = get_skin(skin)
        self._verbose = verbose
        self._total_tokens: int = 0
        self._total_cost: float = 0.0
        self._model_name: str = ""
        self._busy: bool = False
        self._yolo: bool = False
        self._session_id: str = ""

    # ── Compose ─────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        with Horizontal(id="app-container"):
            yield Sidebar()
            with Vertical(id="main-area"):
                with Horizontal(id="header-extra"):
                    yield Static("", id="header-thinking")
                    yield Static("", id="header-spacer")
                    yield Static("", id="header-model")
                yield Transcript()
                with Horizontal(id="status-bar"):
                    yield Static("", id="status-model")
                    yield Static("", id="status-spacer")
                    yield Static("", id="status-tokens")
                    yield Static("", id="status-cost")
                with Container(id="input-area"):
                    yield Static("You >", id="prompt-label")
                    yield InputBar()

    # ── Lifecycle ───────────────────────────────────────────────

    def on_mount(self) -> None:
        transcript = self.query_one(Transcript)
        input_bar = self.query_one(InputBar)
        sidebar = self.query_one(Sidebar)
        input_bar.focus()

        # Apply initial skin
        self._apply_skin(self._skin_name)

        if self._agent:
            mc = getattr(self._agent, "model", None)
            if mc:
                self._model_name = getattr(mc, "model", "?")
                self.query_one("#header-model", Static).update(
                    f"[{self._skin['text3']}]{self._model_name}[/]"
                )
                sidebar.set_model(self._model_name)
                self._session_id = getattr(self._agent, "session_id", "")

        prefix = self._skin.get("agent_prefix", "🤖 Kairos")
        transcript.add_system_msg(f"Welcome to Kairos TUI ✨  [{prefix}]")
        transcript.add_system_msg("Type a message. /help for commands. Ctrl+Q to quit.")
        transcript.add_divider()

        self._refresh_sidebar()

    def _refresh_sidebar(self) -> None:
        sidebar = self.query_one(Sidebar)
        if self._agent and hasattr(self._agent, "list_sessions"):
            try:
                sessions = self._agent.list_sessions()
                sidebar.refresh_sessions(sessions, self._session_id)
            except Exception:
                sidebar.refresh_sessions([])
        try:
            from kairos.tools.registry import get_all_tools
            sidebar.refresh_skills(list(get_all_tools().keys()))
        except Exception:
            sidebar.refresh_skills([])

    # ── Skin ────────────────────────────────────────────────────

    def _apply_skin(self, name: str) -> None:
        """Hot-switch skin CSS at runtime."""
        self._skin_name = name
        self._skin = get_skin(name)
        css = build_skin_css(self._skin)

        # Remove old skin CSS, add new
        if hasattr(self, '_skin_style_id'):
            self.stylesheet.remove(self._skin_style_id)
        self._skin_style_id = self.stylesheet.add(css)

        # Update labels
        self.query_one("#prompt-label", Static).update(
            f"{self._skin.get('user_prefix', 'You')} >"
        )

    # ── Key bindings ────────────────────────────────────────────

    def action_interrupt(self) -> None:
        """Ctrl+C: Interrupt the running agent."""
        if self._agent and hasattr(self._agent, "interrupt"):
            self._agent.interrupt()
        self._set_busy(False)
        self.query_one(Transcript).add_system_msg("⏸ Interrupted")

    def action_clear_screen(self) -> None:
        """Ctrl+L: Clear the transcript."""
        transcript = self.query_one(Transcript)
        transcript.clear_transcript()
        transcript.add_system_msg("Transcript cleared.")

    def action_save_session(self) -> None:
        """Ctrl+S: Save current session."""
        if self._agent and hasattr(self._agent, "save_session"):
            name = self._session_id or "quick-save"
            self._agent.save_session(name)
            self._refresh_sidebar()
            self.query_one(Transcript).add_system_msg(f"Session saved: {name}")

    def action_cancel_input(self) -> None:
        """Escape: Clear input."""
        self.query_one(InputBar).clear()

    # ── Input handling ──────────────────────────────────────────

    def on_input_bar_submitted(self, event: InputBar.Submitted) -> None:
        if event.is_slash:
            self._handle_slash(event.text)
            return
        if not self._agent:
            self._show_error("No agent configured. Set KAIROS_API_KEY.")
            return

        transcript = self.query_one(Transcript)
        prefix = self._skin.get("user_prefix", "You")
        transcript.add_user_msg(event.text)

        self._set_busy(True)
        self.run_worker(self._run_agent(event.text), exclusive=False)

    async def _run_agent(self, message: str) -> None:
        transcript = self.query_one(Transcript)
        status_tokens = self.query_one("#status-tokens", Static)
        status_cost = self.query_one("#status-cost", Static)

        transcript.begin_agent_msg()
        try:
            stream = self._agent.chat_stream(message)
            final_event: dict | None = None
            pending_tools: dict[str, dict] = {}

            for event in stream:
                etype = event.get("type", "")
                if etype == "token":
                    transcript.stream_token(event["content"])
                elif etype == "tool_call":
                    name = event.get("name", "tool")
                    args = event.get("args") or event.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            import json as _json
                            args = _json.loads(args)
                        except Exception:
                            args = {"raw": args}
                    pending_tools[event.get("id", name)] = {
                        "name": name, "args": args,
                        "start_time": time.monotonic(),
                    }
                    transcript.add_tool_card(
                        name=name, args=args if self._verbose else None,
                        status="running",
                    )
                elif etype == "tool_result":
                    name = event.get("name", "tool")
                    result = event.get("result", "")
                    error = event.get("error", "")
                    duration_ms = 0
                    for pt in pending_tools.values():
                        if pt["name"] == name and "result" not in pt:
                            duration_ms = (time.monotonic() - pt["start_time"]) * 1000
                            pt["result"] = result
                            break
                    transcript.add_tool_card(
                        name=name, result=str(result) if result else "",
                        duration_ms=duration_ms,
                        status="error" if error else "done",
                        error=error,
                    )
                elif etype == "done":
                    final_event = event
                elif etype == "error":
                    transcript.add_error_msg(event.get("message", "?"))

            transcript.end_agent_msg()

            if final_event and final_event.get("usage"):
                usage = final_event["usage"]
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total = usage.get("total_tokens", prompt_tokens + completion_tokens)
                cost = (prompt_tokens * 1.5 + completion_tokens * 6.0) / 1_000_000
                self._total_tokens += total
                self._total_cost += cost
                status_tokens.update(f"[{self._skin['text3']}]{self._total_tokens:,} tokens[/]")
                status_cost.update(f"[{self._skin['text3']}]${self._total_cost:.4f}[/]")

        except Exception as e:
            transcript.add_error_msg(f"Agent error: {e}")
        finally:
            self._set_busy(False)
            self._refresh_sidebar()

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        thinking = self.query_one("#header-thinking", Static)
        input_bar = self.query_one(InputBar)
        if busy:
            thinking.update(f"[{self._skin['yellow']}]⏳ thinking…[/]")
            input_bar.disabled = True
            input_bar.placeholder = f"[thinking… Ctrl+C to interrupt]"
        else:
            thinking.update("")
            input_bar.disabled = False
            input_bar.placeholder = "Ask Kairos anything... (/help for commands)"

    # ── Slash commands ──────────────────────────────────────────

    def _handle_slash(self, text: str) -> None:
        parts = text.split()
        cmd = parts[0].lower()
        transcript = self.query_one(Transcript)
        sk = self._skin

        if cmd in ("/exit", "/quit"):
            transcript.add_system_msg("👋 Goodbye!")
            self.exit()

        elif cmd == "/help":
            transcript.add_system_msg(
                "[/]Commands:[/]\n"
                "  /exit /quit  — Exit TUI\n"
                "  /help        — This help\n"
                "  /clear       — Clear transcript\n"
                "  /skin <name> — Switch skin (default/hacker/retro/minimal)\n"
                "  /model <name>— Switch model\n"
                "  /tools       — List available tools\n"
                "  /sessions    — List saved sessions\n"
                "  /save <name> — Save current session\n"
                "  /load <name> — Load a saved session\n"
                "  /verbose     — Toggle verbose tool output\n"
                "  /yolo        — Toggle YOLO mode\n"
                "  /undo        — Undo last exchange\n"
                "\n[/]Keys:[/]\n"
                "  Ctrl+C       — Interrupt agent\n"
                "  Ctrl+L       — Clear screen\n"
                "  Ctrl+S       — Quick save session\n"
                "  Ctrl+Q       — Quit"
            )

        elif cmd == "/clear":
            transcript.clear_transcript()
            transcript.add_system_msg("Transcript cleared.")

        elif cmd == "/verbose":
            self._verbose = not self._verbose
            transcript.add_system_msg(f"Verbose: {'ON' if self._verbose else 'OFF'}")

        elif cmd == "/tools":
            from kairos.tools.registry import get_all_tools
            tools = get_all_tools()
            transcript.add_system_msg(f"Available tools ({len(tools)}):")
            for name in sorted(tools.keys())[:20]:
                transcript.add_system_msg(f"  🔧 {name}")
            if len(tools) > 20:
                transcript.add_system_msg(f"  ... {len(tools) - 20} more")

        elif cmd == "/sessions":
            if self._agent and hasattr(self._agent, "list_sessions"):
                sessions = self._agent.list_sessions()
                if sessions:
                    transcript.add_system_msg(f"Saved sessions ({len(sessions)}):")
                    for s in sessions[:10]:
                        transcript.add_system_msg(
                            f"  📁 {s.get('name','?')} "
                            f"({s.get('turn_count',0)} turns)"
                        )
                else:
                    transcript.add_system_msg("No saved sessions.")
            self._refresh_sidebar()

        elif cmd == "/save" and len(parts) >= 2:
            if self._agent and hasattr(self._agent, "save_session"):
                self._agent.save_session(parts[1])
                transcript.add_system_msg(f"Session saved: {parts[1]}")
                self._refresh_sidebar()

        elif cmd == "/load" and len(parts) >= 2:
            if self._agent and hasattr(self._agent, "load_session"):
                ok = self._agent.load_session(parts[1])
                if ok:
                    transcript.clear_transcript()
                    transcript.add_system_msg(f"Session loaded: {parts[1]}")
                    self._session_id = parts[1]
                    self._refresh_sidebar()
                else:
                    transcript.add_error_msg(f"Session not found: {parts[1]}")

        elif cmd == "/undo":
            if self._agent and hasattr(self._agent, "pop_last_exchange"):
                removed = self._agent.pop_last_exchange()
                transcript.add_system_msg(f"Undid last exchange ({removed} messages).")

        elif cmd == "/skin" and len(parts) >= 2:
            name = parts[1]
            if name in ("default", "hacker", "retro", "minimal"):
                self._apply_skin(name)
                transcript.add_system_msg(f"Skin: {name}")
            else:
                transcript.add_error_msg(
                    f"Unknown skin: {name}. "
                    "Available: default, hacker, retro, minimal"
                )

        elif cmd == "/yolo":
            self._yolo = not self._yolo
            if self._yolo:
                transcript.add_system_msg(
                    "⚡ YOLO mode ON — all safety checks bypassed."
                )
            else:
                transcript.add_system_msg(
                    "🛡 YOLO mode OFF — safety checks restored."
                )

        else:
            transcript.add_system_msg(f"Unknown: {cmd}. /help for commands.")

    def _show_error(self, message: str) -> None:
        self.query_one(Transcript).add_error_msg(message)
