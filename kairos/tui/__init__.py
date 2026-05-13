"""Kairos TUI — Textual-powered terminal user interface.

Provides:
    KairosTUI      — Textual App with multi-panel layout
    Transcript     — Scrollable message display with streaming
    InputBar       — Prompt input with slash commands
    Sidebar        — Session / tools / model panels
"""

from kairos.tui.app import KairosTUI
from kairos.tui.widgets.transcript import Transcript
from kairos.tui.widgets.input_bar import InputBar
from kairos.tui.widgets.sidebar import Sidebar

__all__ = ["KairosTUI", "Transcript", "InputBar", "Sidebar"]
