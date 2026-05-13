"""Kairos TUI — Textual-powered terminal user interface.

Provides:
    KairosTUI      — Textual App (main entry)
    Transcript     — Scrollable message display
    InputBar       — Prompt input with slash commands
"""

from kairos.tui.app import KairosTUI
from kairos.tui.widgets.transcript import Transcript
from kairos.tui.widgets.input_bar import InputBar

__all__ = ["KairosTUI", "Transcript", "InputBar"]
