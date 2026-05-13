"""Textual TUI app for Kairos — multi-panel agent chat interface.

Layout:
    ┌──────────────────────────────────────────────┐
    │  Kairos TUI              thinking… model:xxx │ Header
    ├──────────────────────────────────────────────┤
    │                                              │
    │  Transcript — scrollable messages            │
    │  with streaming and tool cards               │
    │                                              │
    ├──────────────────────────────────────────────┤
    │  model │ 1.2k tokens │ $0.0034              │ Status
    ├──────────────────────────────────────────────┤
    │  You > _                                     │ Input
    └──────────────────────────────────────────────┘

Slash commands: /help /exit /clear /skin /model /tools /sessions /yolo /verbose /save
"""

from __future__ import annotations

import time
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Header, Footer, Static

from kairos.tui.widgets.transcript import Transcript
from kairos.tui.widgets.input_bar import InputBar


class KairosTUI(App):
    """Main Textual App for Kairos agent chat."""

    CSS_PATH = "styles/app.tcss"
    TITLE = "Kairos TUI"
    SUB_TITLE = "The right tool, at the right moment"

    def __init__(
        self,
        agent: Any = None,
        skin: str = "default",
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self._agent = agent
        self._skin = skin
        self._verbose = verbose
        self._total_tokens: int = 0
        self._total_cost: float = 0.0
        self._model_name: str = ""
        self._busy: bool = False

    # ── Compose ─────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)

        # Thinking indicator + model name in header area
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
        input_bar.focus()

        if self._agent:
            mc = getattr(self._agent, "model", None)
            if mc:
                self._model_name = getattr(mc, "model", "?")
                self.query_one("#header-model", Static).update(
                    f"[#8a8f98]{self._model_name}[/]"
                )

        transcript.add_system_msg("Welcome to Kairos TUI  ✨")
        transcript.add_system_msg("Type a message to start. /help for slash commands.")
        transcript.add_divider()

    # ── Input handling ──────────────────────────────────────────

    def on_input_bar_submitted(self, event: InputBar.Submitted) -> None:
        if event.is_slash:
            self._handle_slash(event.text)
            return

        if not self._agent:
            self._show_error("No agent configured. Set KAIROS_API_KEY.")
            return

        transcript = self.query_one(Transcript)
        transcript.add_user_msg(event.text)

        # Mark busy — disables input during agent run
        self._set_busy(True)
        self.run_worker(self._run_agent(event.text), exclusive=False)

    async def _run_agent(self, message: str) -> None:
        """Run the agent and stream output to the transcript."""
        transcript = self.query_one(Transcript)
        status_tokens = self.query_one("#status-tokens", Static)
        status_cost = self.query_one("#status-cost", Static)

        transcript.begin_agent_msg()

        try:
            stream = self._agent.chat_stream(message)
            final_event: dict | None = None
            pending_tools: dict[str, dict] = {}  # tool_call_id → {name, args, start_time}

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

                    tc_id = event.get("id", name)
                    pending_tools[tc_id] = {
                        "name": name, "args": args,
                        "start_time": time.monotonic(),
                    }
                    # Show running immediately
                    transcript.add_tool_card(
                        name=name, args=args if self._verbose else None,
                        status="running",
                    )

                elif etype == "tool_result":
                    name = event.get("name", "tool")
                    result = event.get("result", "")
                    error = event.get("error", "")

                    # Find matching pending tool
                    duration_ms = 0
                    for tid, pt in pending_tools.items():
                        if pt["name"] == name and "result" not in pt:
                            duration_ms = (time.monotonic() - pt["start_time"]) * 1000
                            pt["result"] = result
                            pt["error"] = error
                            break

                    transcript.add_tool_card(
                        name=name,
                        args=None,
                        result=str(result) if result else "",
                        duration_ms=duration_ms,
                        status="error" if error else "done",
                        error=error,
                    )

                elif etype == "done":
                    final_event = event

                elif etype == "error":
                    transcript.add_error_msg(event.get("message", "Unknown error"))

            transcript.end_agent_msg()

            # Update usage
            if final_event and final_event.get("usage"):
                usage = final_event["usage"]
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total = usage.get("total_tokens", prompt_tokens + completion_tokens)
                cost = (prompt_tokens * 1.5 + completion_tokens * 6.0) / 1_000_000

                self._total_tokens += total
                self._total_cost += cost

                status_tokens.update(f"[#8a8f98]{self._total_tokens:,} tokens[/]")
                status_cost.update(f"[#8a8f98]${self._total_cost:.4f}[/]")

        except Exception as e:
            transcript.add_error_msg(f"Agent error: {e}")

        finally:
            self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        """Set busy state — shows/hides thinking indicator, disables input."""
        self._busy = busy
        thinking = self.query_one("#header-thinking", Static)
        input_bar = self.query_one(InputBar)

        if busy:
            thinking.update("[#f5a623]⏳ thinking…[/]")
            input_bar.disabled = True
            input_bar.placeholder = "Kairos is thinking…"
        else:
            thinking.update("")
            input_bar.disabled = False
            input_bar.placeholder = "Ask Kairos anything... (/help for commands)"

    # ── Slash commands ──────────────────────────────────────────

    def _handle_slash(self, text: str) -> None:
        parts = text.split()
        cmd = parts[0].lower()
        transcript = self.query_one(Transcript)

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
                "  /undo        — Undo last exchange"
            )

        elif cmd == "/clear":
            transcript.clear_transcript()
            transcript.add_system_msg("Transcript cleared.")

        elif cmd == "/verbose":
            self._verbose = not self._verbose
            transcript.add_system_msg(
                f"Verbose: {'ON' if self._verbose else 'OFF'}"
            )

        elif cmd == "/tools":
            if self._agent:
                from kairos.tools.registry import get_all_tools
                tools = get_all_tools()
                transcript.add_system_msg(f"Available tools ({len(tools)}):")
                for name in sorted(tools.keys())[:20]:
                    transcript.add_system_msg(f"  🔧 {name}")
                if len(tools) > 20:
                    transcript.add_system_msg(f"  ... and {len(tools) - 20} more")
            else:
                transcript.add_system_msg("No agent loaded.")

        elif cmd == "/sessions":
            if self._agent and hasattr(self._agent, "list_sessions"):
                sessions = self._agent.list_sessions()
                if sessions:
                    transcript.add_system_msg(f"Saved sessions ({len(sessions)}):")
                    for s in sessions[:10]:
                        transcript.add_system_msg(
                            f"  📁 {s.get('name', '?')} "
                            f"({s.get('turn_count', 0)} turns)"
                        )
                else:
                    transcript.add_system_msg("No saved sessions.")
            else:
                transcript.add_system_msg("Session listing not available.")

        elif cmd == "/save" and len(parts) >= 2:
            if self._agent and hasattr(self._agent, "save_session"):
                self._agent.save_session(parts[1])
                transcript.add_system_msg(f"Session saved: {parts[1]}")
            else:
                transcript.add_system_msg("Session saving not available.")

        elif cmd == "/load" and len(parts) >= 2:
            if self._agent and hasattr(self._agent, "load_session"):
                ok = self._agent.load_session(parts[1])
                if ok:
                    transcript.clear_transcript()
                    transcript.add_system_msg(f"Session loaded: {parts[1]}")
                else:
                    transcript.add_error_msg(f"Session not found: {parts[1]}")
            else:
                transcript.add_system_msg("Session loading not available.")

        elif cmd == "/undo":
            if self._agent and hasattr(self._agent, "pop_last_exchange"):
                removed = self._agent.pop_last_exchange()
                transcript.add_system_msg(f"Undid last exchange ({removed} messages).")
            else:
                transcript.add_system_msg("Undo not available.")

        elif cmd == "/skin" and len(parts) >= 2:
            skin_name = parts[1]
            if skin_name in ("default", "hacker", "retro", "minimal"):
                self._skin = skin_name
                transcript.add_system_msg(f"Skin: {skin_name}")
            else:
                transcript.add_system_msg(
                    f"Unknown skin: {skin_name}. "
                    "Available: default, hacker, retro, minimal"
                )

        elif cmd == "/yolo":
            transcript.add_system_msg("⚡ YOLO mode toggled (TUI)")

        else:
            transcript.add_system_msg(f"Unknown command: {cmd}. /help for commands.")

    def _show_error(self, message: str) -> None:
        self.query_one(Transcript).add_error_msg(message)
