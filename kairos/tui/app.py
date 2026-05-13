"""Textual TUI app for Kairos — multi-panel agent chat interface.

Layout:
    ┌──────────────────────────────────────────────┐
    │  Kairos TUI                   model: xxx     │ Header
    ├──────────────────────────────────────────────┤
    │                                              │
    │  Transcript — scrollable messages            │
    │                                              │
    ├──────────────────────────────────────────────┤
    │  model │ 1.2k tokens │ $0.0034              │ Status
    ├──────────────────────────────────────────────┤
    │  You ❯ _                                     │ Input
    └──────────────────────────────────────────────┘

Slash commands: /help /exit /clear /skin /model /tools /sessions /yolo /verbose /save
"""

from __future__ import annotations

import asyncio
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
        self._running: bool = True

    # ── Compose ─────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        """Build the UI layout."""
        yield Header(show_clock=False)
        yield Transcript()
        with Horizontal(id="status-bar"):
            yield Static("", id="status-model")
            yield Static("", id="status-spacer")
            yield Static("", id="status-tokens")
            yield Static("", id="status-cost")
        with Container(id="input-area"):
            yield Static("You ❯", id="prompt-label")
            yield InputBar()

    # ── Lifecycle ───────────────────────────────────────────────

    def on_mount(self) -> None:
        """Called when app is ready."""
        transcript = self.query_one(Transcript)
        input_bar = self.query_one(InputBar)

        # Focus input
        input_bar.focus()

        # Set model name in header
        if self._agent:
            mc = getattr(self._agent, "model", None)
            if mc:
                self._model_name = getattr(mc, "model", "?")
                self.query_one("#status-model", Static).update(
                    f"[#5e6ad2]{self._model_name}[/]"
                )

        # Welcome
        transcript.add_system_msg("Welcome to Kairos TUI ✨")
        transcript.add_system_msg(
            "Type a message to start. /help for slash commands."
        )
        transcript.add_divider()

    # ── Input handling ──────────────────────────────────────────

    def on_input_bar_submitted(self, event: InputBar.Submitted) -> None:
        """Handle user message submission."""
        if event.is_slash:
            self._handle_slash(event.text)
            return

        if not self._agent:
            self._show_error("No agent configured. Set KAIROS_API_KEY.")
            return

        # Show user message
        transcript = self.query_one(Transcript)
        transcript.add_user_msg(event.text)

        # Run agent in worker
        self.run_worker(
            self._run_agent(event.text),
            exclusive=False,
        )

    async def _run_agent(self, message: str) -> None:
        """Run the agent and stream output to the transcript."""
        transcript = self.query_one(Transcript)
        status_model = self.query_one("#status-model", Static)
        status_tokens = self.query_one("#status-tokens", Static)
        status_cost = self.query_one("#status-cost", Static)

        transcript.begin_agent_msg()

        try:
            stream = self._agent.chat_stream(message)

            final_event: dict | None = None
            for event in stream:
                if event["type"] == "token":
                    transcript.stream_token(event["content"])

                elif event["type"] == "tool_call":
                    # Flush pending text, then show tool
                    name = event.get("name", "tool")
                    transcript.add_tool_call(name)

                elif event["type"] == "tool_result":
                    name = event.get("name", "tool")
                    result = event.get("result", "")
                    transcript.add_tool_call(
                        name, result=str(result),
                        duration_ms=event.get("duration_ms", 0),
                    )

                elif event["type"] == "done":
                    final_event = event

                elif event["type"] == "error":
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

                status_tokens.update(
                    f"[#8a8f98]{self._total_tokens:,} tokens[/]"
                )
                status_cost.update(
                    f"[#8a8f98]${self._total_cost:.4f}[/]"
                )

        except Exception as e:
            transcript.add_error_msg(f"Agent error: {e}")

    # ── Slash commands ──────────────────────────────────────────

    def _handle_slash(self, text: str) -> None:
        """Dispatch slash commands."""
        parts = text.split()
        cmd = parts[0].lower()
        transcript = self.query_one(Transcript)

        if cmd in ("/exit", "/quit"):
            transcript.add_system_msg("👋 Goodbye!")
            self.exit()

        elif cmd == "/help":
            transcript.add_system_msg(
                "[/]Commands:[/]\n"
                "  /exit /quit  — Exit\n"
                "  /help        — This help\n"
                "  /clear       — Clear transcript\n"
                "  /skin <name> — Switch theme\n"
                "  /model <name>— Switch model\n"
                "  /tools       — List tools\n"
                "  /sessions    — List sessions\n"
                "  /save <name> — Save session\n"
                "  /verbose     — Toggle verbose\n"
                "  /yolo        — Toggle YOLO mode"
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
                    transcript.add_system_msg(f"  ... and {len(tools)-20} more")
            else:
                transcript.add_system_msg("No agent loaded.")

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
            else:
                transcript.add_system_msg("Session listing not available.")

        elif cmd == "/save" and len(parts) >= 2:
            if self._agent and hasattr(self._agent, "save_session"):
                self._agent.save_session(parts[1])
                transcript.add_system_msg(f"Session saved: {parts[1]}")
            else:
                transcript.add_system_msg("Session saving not available.")

        elif cmd == "/skin" and len(parts) >= 2:
            skin_name = parts[1]
            if skin_name in ("default", "hacker", "retro", "minimal"):
                self._skin = skin_name
                transcript.add_system_msg(f"Skin: {skin_name}")
                # TODO: apply skin colors to CSS dynamically
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
        """Show error in transcript."""
        self.query_one(Transcript).add_error_msg(message)
