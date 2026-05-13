"""Input bar widget for Kairos TUI.

Features:
  - Prompt prefix ("You ❯")
  - Slash command detection (/command)
  - Keyboard: Enter to send, Escape to clear
  - Multi-line support (Shift+Enter or auto-expand)
"""

from __future__ import annotations

from textual.widgets import Input
from textual import events
from textual.message import Message


class InputBar(Input):
    """Custom input bar with prompt styling and slash command support."""

    class Submitted(Message):
        """Emitted when user presses Enter with non-empty input."""

        def __init__(self, text: str, is_slash: bool = False) -> None:
            super().__init__()
            self.text = text.strip()
            self.is_slash = is_slash

    class SlashCommand(Message):
        """Emitted when user types a slash command."""

        def __init__(self, command: str) -> None:
            super().__init__()
            self.command = command.strip()

    def __init__(self) -> None:
        super().__init__(
            placeholder="Ask Kairos anything... (/help for commands)",
            id="input-bar",
        )

    def on_key(self, event: events.Key) -> None:
        """Handle special keys."""
        # Enter → submit
        if event.key == "enter":
            text = self.value.strip()
            if text:
                event.prevent_default()
                is_slash = text.startswith("/")
                self.clear()
                self.post_message(self.Submitted(text, is_slash=is_slash))
                if is_slash:
                    self.post_message(self.SlashCommand(text))
            return

        # Escape → clear
        if event.key == "escape":
            self.clear()
            return
