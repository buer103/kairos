"""Transcript widget — scrollable message display for Kairos TUI.

Supports:
  - User / agent / system / error message styling
  - Tool call cards with arguments, results, and timing
  - Token-by-token streaming with throttled refresh
  - Auto-scroll to bottom
  - Rich markup for colors and code blocks
"""

from __future__ import annotations

import time
from textual.widgets import RichLog


class Transcript(RichLog):
    """Scrollable transcript with streaming and tool call support.

    Uses RichLog for built-in scroll, markup, and auto-scroll.
    Streaming tokens are accumulated and flushed every ~50ms for smooth rendering.
    """

    def __init__(self) -> None:
        super().__init__(
            markup=True,
            highlight=True,
            wrap=True,
            min_width=40,
            id="transcript",
        )
        self._streaming: bool = False
        self._stream_buffer: str = ""
        self._stream_last_flush: float = 0.0
        self._stream_flush_interval: float = 0.05  # 50ms flush interval

    # ── Public API ──────────────────────────────────────────────

    def add_user_msg(self, text: str) -> None:
        """Display user message."""
        self.write(f"\n[bold #5e6ad2]You[/]  {text}\n")

    def begin_agent_msg(self) -> None:
        """Start an agent response — prepares for streaming."""
        self.write("\n[bold #d0d6e0]🤖 Kairos[/]  ")
        self._streaming = True
        self._stream_buffer = ""
        self._stream_last_flush = time.monotonic()

    def stream_token(self, token: str) -> None:
        """Append a token. Flushes to RichLog every ~50ms for smooth rendering."""
        self._stream_buffer += token
        now = time.monotonic()
        if now - self._stream_last_flush >= self._stream_flush_interval:
            self._flush_stream()

    def _flush_stream(self) -> None:
        """Write buffered tokens to RichLog."""
        if self._stream_buffer:
            self.write(self._stream_buffer, scroll_end=True)
            self._stream_buffer = ""
            self._stream_last_flush = time.monotonic()

    def end_agent_msg(self) -> None:
        """Finalize the agent message."""
        self._flush_stream()
        self._streaming = False
        self.write("\n")

    def add_tool_card(self, name: str, args: dict | None = None,
                      result: str = "", duration_ms: float = 0,
                      status: str = "done", error: str = "") -> None:
        """Display a tool call as a styled card.

        Shows tool name, arguments (collapsed), result preview, and timing.
        """
        self._flush_stream()

        icon = _TOOL_ICONS.get(name, "🔧")
        status_icon = {"running": "⏳", "done": "✅", "error": "❌"}.get(status, "🔧")
        time_str = f" {duration_ms:.0f}ms" if duration_ms else ""

        # Header line
        self.write(
            f"  [bold #f5a623]{status_icon} {icon} {name}[/]"
            f"[#62666d]{time_str}[/]\n"
        )

        # Arguments (verbose or always for key tools)
        if args:
            args_preview = _format_args(args)
            if args_preview:
                self.write(f"    [#8a8f98]args:[/] {args_preview}\n")

        # Result
        if result:
            result_preview = _format_result(result)
            if result_preview:
                self.write(f"    [#8a8f98]→[/] {result_preview}\n")

        # Error
        if error:
            self.write(f"    [#e5484d]✖ {error}[/]\n")

    def add_system_msg(self, text: str) -> None:
        """Display a system/info message."""
        self.write(f"[#62666d italic]{text}[/]\n")

    def add_error_msg(self, text: str) -> None:
        """Display an error message."""
        self.write(f"[bold #e5484d]⚠ {text}[/]\n")

    def add_divider(self) -> None:
        """Add a visual divider."""
        self.write("─" * 50 + "\n")

    def clear_transcript(self) -> None:
        """Clear all messages."""
        self.clear()
        self._streaming = False
        self._stream_buffer = ""


# ═══════════════════════════════════════════════════════════════
# Tool display helpers
# ═══════════════════════════════════════════════════════════════

_TOOL_ICONS: dict[str, str] = {
    "read_file": "📖",
    "search_files": "🔍",
    "write_file": "✏️",
    "patch": "🔧",
    "terminal": "💻",
    "execute_code": "🐍",
    "web_search": "🌐",
    "web_fetch": "📥",
    "delegate_task": "🤖",
    "skill_view": "📚",
    "skill_manage": "📝",
    "browser": "🖥️",
    "vision_analyze": "👁️",
    "memory": "🧠",
    "session_search": "📅",
    "cronjob": "⏰",
    "send_message": "📤",
    "clarify": "❓",
    "process": "⚙️",
    "todo": "📋",
}


def _format_args(args: dict) -> str:
    """Format tool arguments for display. Returns empty string if empty."""
    if not args:
        return ""

    # Pick most interesting keys
    key_args = {}
    for k in ("command", "query", "pattern", "path", "message", "name", "goal",
              "action", "question", "content", "text", "prompt"):
        if k in args:
            val = str(args[k])
            if len(val) > 80:
                val = val[:80] + "..."
            key_args[k] = val

    if not key_args:
        # Show first 2 keys
        for k, v in list(args.items())[:2]:
            val = str(v)
            if len(val) > 60:
                val = val[:60] + "..."
            key_args[k] = val

    pairs = [f"[#d0d6e0]{k}[/]=[#8a8f98]\"{v}\"[/]" for k, v in key_args.items()]
    return ", ".join(pairs) if pairs else ""


def _format_result(result: str) -> str:
    """Format tool result for display. Truncates long results."""
    if not result:
        return ""

    text = str(result).strip()
    # Truncate long results to a few lines
    lines = text.split("\n")
    if len(lines) > 5:
        lines = lines[:5]
        text = "\n".join(lines) + f"\n    [#62666d]... ({len(result.splitlines())} total lines)[/]"
    elif len(text) > 500:
        text = text[:500] + f"... [#62666d]({len(result)} chars)[/]"

    return f"[#8a8f98]{text}[/]"
