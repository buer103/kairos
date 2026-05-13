"""Transcript widget — scrollable message display for Kairos TUI.

Supports:
  - User / agent / system / error message styling
  - Tool call inline display
  - Token-by-token streaming (append to last message)
  - Auto-scroll to bottom
  - Rich markup for colors
"""

from __future__ import annotations

from textual.widgets import RichLog
from textual.widget import Widget


class Transcript(RichLog):
    """Scrollable transcript displaying the conversation.

    Uses RichLog for built-in scroll, markup, and auto-scroll.
    Writing is append-only — the widget manages scroll state internally.
    """

    def __init__(self) -> None:
        super().__init__(
            markup=True,
            highlight=True,
            wrap=True,
            min_width=40,
            id="transcript",
        )
        self._streaming_line: int | None = None
        self._streaming_buffer: str = ""

    # ── Public API ──────────────────────────────────────────────

    def add_user_msg(self, text: str) -> None:
        """Display user message."""
        self.write(f"[user-msg]You:[/] {text}\n")

    def begin_agent_msg(self) -> None:
        """Start an agent response — prepares for streaming."""
        self.write("[agent-msg]🤖 Kairos:[/] ", scroll_end=True)
        self._streaming_line = None
        self._streaming_buffer = ""
        # Store reference to the line we'll stream into
        # RichLog doesn't expose line references, so we use a different approach:
        # We'll use `write(..., scroll_end=True)` and simply append.

    def stream_token(self, token: str) -> None:
        """Append a token to the current agent message (no newline)."""
        self.write(token, scroll_end=True)

    def end_agent_msg(self) -> None:
        """Finalize the agent message."""
        self.write("\n\n", scroll_end=True)
        self._streaming_buffer = ""

    def add_tool_call(self, name: str, result: str = "", duration_ms: float = 0) -> None:
        """Display a tool call."""
        icon = "🔧"
        time_str = f" ({duration_ms:.0f}ms)" if duration_ms else ""
        self.write(f"  [tool-header]{icon} {name}[/]{time_str}")
        if result:
            result_preview = str(result)[:200]
            if len(str(result)) > 200:
                result_preview += "..."
            self.write(f" [tool-result]→ {result_preview}[/]")
        self.write("\n")

    def add_system_msg(self, text: str) -> None:
        """Display a system/info message."""
        self.write(f"[system-msg]{text}[/]\n")

    def add_error_msg(self, text: str) -> None:
        """Display an error message."""
        self.write(f"[error-msg]⚠ {text}[/]\n")

    def add_divider(self) -> None:
        """Add a visual divider."""
        self.write("─" * 40 + "\n")

    def clear_transcript(self) -> None:
        """Clear all messages."""
        self.clear()
        self._streaming_buffer = ""
