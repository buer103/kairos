"""Rich-powered terminal UI for Kairos.

Features:
  - Rich panels for agent output
  - Spinner while LLM is thinking
  - Skin / theme system (light, dark, hacker, retro)
  - Slash commands: /exit, /help, /history, /clear, /model, /tools, /verbose
  - Tool call visualization with tree rendering
  - Streaming output support
"""

from __future__ import annotations

import os
import sys
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from rich.style import Style
from rich.live import Live
from rich.syntax import Syntax
from rich.prompt import Prompt
from rich.box import Box, ROUNDED, HEAVY, SIMPLE, MINIMAL

# ═══════════════════════════════════════════════════════════════
# Skins / Themes
# ═══════════════════════════════════════════════════════════════

SKINS = {
    "default": {
        "agent_color": "cyan",
        "user_color": "green",
        "tool_color": "yellow",
        "error_color": "red",
        "info_color": "blue",
        "box": ROUNDED,
        "spinner": "dots",
    },
    "hacker": {
        "agent_color": "bright_green",
        "user_color": "green",
        "tool_color": "bright_yellow",
        "error_color": "bright_red",
        "info_color": "cyan",
        "box": MINIMAL,
        "spinner": "line",
    },
    "retro": {
        "agent_color": "bright_cyan",
        "user_color": "bright_magenta",
        "tool_color": "bright_yellow",
        "error_color": "red",
        "info_color": "white",
        "box": SIMPLE,
        "spinner": "arc",
    },
    "minimal": {
        "agent_color": "white",
        "user_color": "dim white",
        "tool_color": "dim yellow",
        "error_color": "red",
        "info_color": "dim blue",
        "box": MINIMAL,
        "spinner": "dots",
    },
}


class KairosConsole:
    """Rich-powered console for Kairos agent interactions."""

    def __init__(
        self,
        skin: str = "default",
        verbose: bool = False,
        stream: bool = True,
    ):
        self.console = Console()
        self.skin_name = skin
        self.skin = SKINS.get(skin, SKINS["default"])
        self.verbose = verbose
        self.stream = stream
        self._history: list[dict[str, Any]] = []
        self._spinner: Spinner | None = None
        self._live: Live | None = None

    # ═══════════════════════════════════════════════════════════
    # Output helpers
    # ═══════════════════════════════════════════════════════════

    def agent_output(self, content: str, confidence: float | None = None) -> None:
        """Display agent response in a styled panel."""
        md = Markdown(content)
        title_parts = [Text("🤖 Kairos", style=self.skin["agent_color"])]
        if confidence is not None:
            title_parts.append(Text(f"  [conf={confidence:.2f}]", style="dim"))
        panel = Panel(
            md,
            title=Text.assemble(*title_parts),
            border_style=self.skin["agent_color"],
            box=self.skin["box"],
        )
        self.console.print(panel)
        self._history.append({"role": "agent", "content": content, "confidence": confidence})

    def user_input(self, content: str) -> None:
        """Display user input in a styled panel."""
        panel = Panel(
            content,
            title=Text("You", style=self.skin["user_color"]),
            border_style=self.skin["user_color"],
            box=self.skin["box"],
        )
        self.console.print(panel)
        self._history.append({"role": "user", "content": content})

    def tool_call(self, name: str, args: dict[str, Any], result: Any, duration_ms: float = 0) -> None:
        """Display a tool call with its arguments and result."""
        if not self.verbose:
            return

        tree = Tree(
            f"[{self.skin['tool_color']}]🔧 {name}[/]  ({duration_ms:.0f}ms)",
            guide_style="dim",
        )
        args_branch = tree.add("[dim]args[/]")
        for k, v in args.items():
            v_str = str(v)
            if len(v_str) > 120:
                v_str = v_str[:120] + "..."
            args_branch.add(f"[dim]{k}:[/] {v_str}")

        result_str = str(result)
        if len(result_str) > 300:
            result_str = result_str[:300] + "..."
        tree.add(f"[dim]result:[/] {result_str}")

        self.console.print(tree)
        self._history.append({
            "role": "tool",
            "name": name,
            "args": args,
            "result": result,
            "duration_ms": duration_ms,
        })

    def error(self, message: str) -> None:
        """Display an error message."""
        panel = Panel(
            message,
            title=Text("⚠️ Error", style="bold red"),
            border_style="red",
            box=self.skin["box"],
        )
        self.console.print(panel)

    def info(self, message: str) -> None:
        """Display an info message."""
        self.console.print(f"[{self.skin['info_color']}]ℹ️  {message}[/]")

    def success(self, message: str) -> None:
        """Display a success message."""
        self.console.print(f"[green]✅ {message}[/]")

    def spinner_start(self, message: str = "Thinking...") -> None:
        """Start a spinner to indicate processing."""
        self._spinner = Spinner(self.skin["spinner"], text=message, style=self.skin["agent_color"])
        self._live = Live(self._spinner, console=self.console, refresh_per_second=10)
        self._live.start()

    def spinner_stop(self) -> None:
        """Stop the spinner."""
        if self._live:
            self._live.stop()
            self._live = None
        self._spinner = None

    def spinner_update(self, message: str) -> None:
        """Update the spinner message."""
        if self._spinner:
            self._spinner.text = message
            if self._live:
                self._live.update(self._spinner)

    # ═══════════════════════════════════════════════════════════
    # Display helpers
    # ═══════════════════════════════════════════════════════════

    def show_help(self) -> None:
        """Display available slash commands."""
        table = Table(
            title="Kairos Slash Commands",
            border_style=self.skin["info_color"],
            box=self.skin["box"],
        )
        table.add_column("Command", style="bold cyan")
        table.add_column("Description", style="dim")

        commands = [
            ("/exit, /quit", "Exit the chat session"),
            ("/help", "Show this help"),
            ("/history", "Show conversation history summary"),
            ("/clear", "Clear conversation history"),
            ("/model <name>", "Switch model (e.g., /model gpt-4)"),
            ("/skin <name>", "Switch skin (default/hacker/retro/minimal)"),
            ("/tools", "List available tools"),
            ("/verbose", "Toggle verbose tool output"),
            ("/cron list", "List scheduled cron jobs"),
            ("/run <query>", "Run a one-shot query"),
            ("/save <name>", "Save current session"),
            ("/sessions", "List saved sessions"),
            ("/perm [show|trust|ask|block <tool>]", "Manage permission policies"),
            ("/skills", "List installed skills"),
        ]
        for cmd, desc in commands:
            table.add_row(cmd, desc)

        self.console.print(table)

    def show_history(self) -> None:
        """Display conversation history summary."""
        if not self._history:
            self.info("No conversation history yet.")
            return

        table = Table(
            title=f"Conversation History ({len(self._history)} messages)",
            border_style=self.skin["info_color"],
            box=self.skin["box"],
        )
        table.add_column("#", style="dim")
        table.add_column("Role", style="bold")
        table.add_column("Preview")

        for i, entry in enumerate(self._history):
            role = entry["role"]
            preview = str(entry.get("content", entry.get("name", "")))[:80]
            color = {
                "user": self.skin["user_color"],
                "agent": self.skin["agent_color"],
                "tool": self.skin["tool_color"],
            }.get(role, "white")
            table.add_row(str(i + 1), f"[{color}]{role}[/]", preview)

        self.console.print(table)

    def show_tools(self, tools: list[dict[str, Any]] | None = None) -> None:
        """Display available tools."""
        from kairos.tools.registry import get_all_tools

        tools = tools or get_all_tools()
        if not tools:
            self.info("No tools registered.")
            return

        table = Table(
            title=f"Available Tools ({len(tools)})",
            border_style=self.skin["info_color"],
            box=self.skin["box"],
        )
        table.add_column("Tool", style="bold cyan")
        table.add_column("Description", style="dim")
        table.add_column("Params", style="dim")

        for t in tools:
            name = t.get("name", "?")
            desc = (t.get("description", "") or "")[:60]
            params = ", ".join(t.get("parameters", {}).keys())[:50]
            table.add_row(name, desc, params)

        self.console.print(table)

    def show_cron_jobs(self, jobs: list[Any]) -> None:
        """Display scheduled cron jobs."""
        if not jobs:
            self.info("No cron jobs registered.")
            return

        table = Table(
            title=f"Cron Jobs ({len(jobs)})",
            border_style=self.skin["info_color"],
            box=self.skin["box"],
        )
        table.add_column("ID", style="dim")
        table.add_column("Name", style="bold")
        table.add_column("Status", style="bold")
        table.add_column("Next Run")
        table.add_column("Runs")

        for j in jobs:
            d = j.to_dict() if hasattr(j, "to_dict") else j
            status_color = {
                "pending": "yellow",
                "running": "cyan",
                "done": "green",
                "error": "red",
                "paused": "dim",
                "cancelled": "dim",
            }.get(d.get("status", ""), "white")
            table.add_row(
                d.get("id", "")[:12],
                d.get("name", ""),
                f"[{status_color}]{d.get('status', '?')}[/]",
                d.get("next_run", "—")[:19],
                str(d.get("run_count", 0)),
            )

        self.console.print(table)

    # ═══════════════════════════════════════════════════════════
    # Input
    # ═══════════════════════════════════════════════════════════

    def prompt(self, prompt_text: str = "> ") -> str:
        """Prompt for user input with Rich styling."""
        return Prompt.ask(f"[{self.skin['user_color']}]{prompt_text}[/]")

    def set_skin(self, name: str) -> bool:
        """Switch to a different skin."""
        if name in SKINS:
            self.skin_name = name
            self.skin = SKINS[name]
            self.success(f"Switched to skin: {name}")
            return True
        else:
            self.error(f"Unknown skin: {name}. Available: {', '.join(SKINS.keys())}")
            return False
