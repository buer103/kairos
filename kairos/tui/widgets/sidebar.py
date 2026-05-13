"""Sidebar widget for Kairos TUI.

Panels:
  - Sessions: list, switch, save, delete
  - Skills: list installed skills
  - Model: current model, switch via /model

Uses Textual's Tree widget for collapsible sections.
"""

from __future__ import annotations

from textual.widgets import Static, Tree
from textual.widgets.tree import TreeNode
from textual.containers import Vertical
from textual.app import ComposeResult


class Sidebar(Vertical):
    """Left sidebar with session/skill/model panels."""

    def __init__(self) -> None:
        super().__init__(id="sidebar")
        self._session_names: list[str] = []
        self._skill_names: list[str] = []
        self._model_name: str = ""

    def compose(self) -> ComposeResult:
        yield Static("🔵 Kairos", id="sidebar-brand")
        yield Tree("Sessions", id="sidebar-sessions")
        yield Tree("Tools", id="sidebar-tools")
        yield Static("", id="sidebar-model")

    def on_mount(self) -> None:
        """Hide trees initially — populated on first data load."""
        pass

    # ── Refresh ─────────────────────────────────────────────────

    def refresh_sessions(self, sessions: list[dict],
                         active_session: str = "") -> None:
        """Populate the Sessions tree."""
        tree = self.query_one("#sidebar-sessions", Tree)
        tree.clear()
        tree.label = f"📁 Sessions ({len(sessions)})"

        if not sessions:
            tree.root.add("(none)")
            return

        for s in sessions[:20]:
            name = s.get("name", "?")
            turns = s.get("turn_count", 0)
            label = f"{name}"
            node = tree.root.add(label)
            node.data = name
            if name == active_session:
                node.label = f"● {name}"
            if turns:
                node = tree.root.children[-1]
                node.label += f"  [{turns}t]"

        if len(sessions) > 20:
            tree.root.add(f"... and {len(sessions) - 20} more")

    def refresh_skills(self, skills: list[str]) -> None:
        """Populate the Skills tree."""
        tree = self.query_one("#sidebar-tools", Tree)
        tree.clear()
        tree.label = f"🔧 Tools ({len(skills)})"

        if not skills:
            tree.root.add("(none)")
            return

        for name in sorted(skills)[:20]:
            tree.root.add(name)

        if len(skills) > 20:
            tree.root.add(f"... {len(skills) - 20} more")

    def set_model(self, name: str) -> None:
        """Update the model display."""
        self._model_name = name
        self.query_one("#sidebar-model", Static).update(
            f"[#8a8f98]🤖 {name}[/]"
        )
