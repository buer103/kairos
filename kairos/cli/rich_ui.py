"""Rich-powered terminal UI for Kairos.

Features:
  - Rich panels for agent output
  - **Live streaming output** — token-by-token rendering via Rich Live
  - Spinner while LLM is thinking
  - Skin / theme system (10 skins via SkinEngine)
  - Slash commands: /exit, /help, /history, /clear, /model, /tools, /verbose
  - Tool call visualization with tree rendering
  - **Status bar** — model, session, token usage
  - **Token usage + cost tracking**
  - Markdown with syntax-highlighted code blocks
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Generator

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
from rich.align import Align


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
    """Rich-powered console for Kairos agent interactions.

    Supports streaming output via :meth:`stream_response` for real-time
    token-by-token rendering using Rich Live.
    """

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

        # Status bar info
        self._status_model: str = ""
        self._status_session: str = ""
        self._total_tokens: int = 0
        self._total_cost: float = 0.0

        # Safety / UX toggles
        self._yolo_mode: bool = False  # /yolo — skip dangerous command checks

    # ═══════════════════════════════════════════════════════════
    # Output helpers
    # ═══════════════════════════════════════════════════════════

    def agent_output(self, content: str, confidence: float | None = None) -> None:
        """Display agent response in a styled panel."""
        md = Markdown(content, code_theme="monokai")
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
        """Display a tool call. Always shows one-line indicator; tree in verbose."""
        # Always show a one-liner
        result_str = str(result)
        if len(result_str) > 80:
            result_str = result_str[:80] + "..."
        self.console.print(
            f"  [dim]🔧 {name}[/] [dim]{result_str}[/]"
        )

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

    def error(self, message: str, hint: str = "") -> None:
        """Display an error with optional fix suggestion."""
        content = Text(message, style="bold red")
        if hint:
            content = Text.assemble(
                (message + "\n", "bold red"),
                ("  💡 ", "dim"),
                (hint, "dim"),
            )
        panel = Panel(
            content,
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
    # Streaming output
    # ═══════════════════════════════════════════════════════════

    def stream_response(
        self,
        stream: Generator[dict, None, None],
        agent_name: str = "Kairos",
    ) -> tuple[str, dict | None]:
        """Render streaming agent output token-by-token.

        Tokens are printed immediately with console.print(token, end="").
        No Rich Live — avoids Windows terminal rendering bugs.
        Tool calls insert separator lines. Final output wrapped in Panel.

        Returns:
            (final_content, final_event)
        """
        content_blocks: list[str] = []
        current_block: list[str] = []
        tool_call_names: list[str] = []
        final_event: dict | None = None

        def _assemble() -> str:
            parts = []
            for i, block in enumerate(content_blocks):
                if i > 0 and i - 1 < len(tool_call_names):
                    parts.append(f"\n\n---\n*🔧 `{tool_call_names[i - 1]}`*\n")
                parts.append(block)
            if current_block:
                parts.append("".join(current_block))
            return "".join(parts)

        if not self.stream:
            # Non-streaming: collect, render once
            for event in stream:
                if event["type"] == "token":
                    current_block.append(event["content"])
                elif event["type"] == "tool_call":
                    tool_call_names.append(event.get("name", "tool"))
                elif event["type"] == "done":
                    if current_block:
                        content_blocks.append("".join(current_block))
                        current_block = []
                    final_event = event
            final_content = _assemble()
            self.agent_output(final_content)
            return final_content, final_event

        # ── Streaming: print tokens directly, no Live ──────────
        # Print panel header
        self.console.print()
        header = Panel(
            Text(""),
            title=Text(f"🤖 {agent_name}", style=self.skin["agent_color"]),
            border_style=self.skin["agent_color"],
            box=self.skin["box"],
            padding=(0, 2),
        )
        self.console.print(header)

        first_token = True
        for event in stream:
            if event["type"] == "token":
                token = event["content"]
                current_block.append(token)
                # Print token inline — no newline between tokens
                self.console.print(token, end="", highlight=False)
                first_token = False

            elif event["type"] == "tool_call":
                tool_call_names.append(event.get("name", "tool"))
                if current_block:
                    content_blocks.append("".join(current_block))
                    current_block = []
                self.console.print()
                self.console.print(
                    f"[yellow]🔧 Calling `{event.get('name', 'tool')}`...[/]"
                )

            elif event["type"] == "done":
                if current_block:
                    content_blocks.append("".join(current_block))
                    current_block = []
                final_event = event

        self.console.print()
        self.console.print()

        final_content = _assemble()
        # Render final markdown in a proper panel
        self.agent_output(final_content)
        self._history.append({"role": "agent", "content": final_content, "streaming": True})
        return final_content, final_event

    # ═══════════════════════════════════════════════════════════
    # Status bar
    # ═══════════════════════════════════════════════════════════

    def set_status(self, *, model: str = "", session: str = "", tokens: int = 0, cost: float = 0.0) -> None:
        """Update status bar info."""
        if model:
            self._status_model = model
        if session:
            self._status_session = session
        if tokens:
            self._total_tokens = tokens
        if cost:
            self._total_cost = cost

    def add_tokens(self, tokens: int, cost: float = 0.0) -> None:
        """Increment token and cost counters."""
        self._total_tokens += tokens
        self._total_cost += cost

    # ═══════════════════════════════════════════════════════════
    # Token usage display
    # ═══════════════════════════════════════════════════════════

    def show_usage(self, usage: dict[str, Any]) -> None:
        """Display token usage from a provider 'done' event."""
        if not usage:
            return
        prompt = usage.get("prompt_tokens", 0)
        completion = usage.get("completion_tokens", 0)
        total = usage.get("total_tokens", prompt + completion)

        parts = [f"[dim]📊 Tokens:[/] {total:,}"]
        if prompt:
            parts.append(f"[dim]in:[/] {prompt:,}")
        if completion:
            parts.append(f"[dim]out:[/] {completion:,}")

        # Estimate cost
        if total > 0:
            # Rough estimates per 1K tokens (configurable later)
            cost = (prompt * 1.5 + completion * 6.0) / 1_000_000  # ~GPT-4o pricing
            self.add_tokens(total, cost)
            parts.append(f"[dim]≈ ${cost:.4f}[/]")

        self.console.print(Text("  ".join(parts)))
        self.console.print()

    # ═══════════════════════════════════════════════════════════
    # Display helpers
    # ═══════════════════════════════════════════════════════════

    def show_welcome(self, version: str, model: str, base_url: str,
                     session_count: int, tool_count: int) -> None:
        """First-run welcome panel showing capabilities at a glance."""
        panel = Panel(
            Text.assemble(
                ("Welcome to Kairos ", "bold cyan"),
                (f"v{version}\n\n", "dim"),
                ("You're connected to ", ""),
                (f"{model}", "bold"),
                (f"\n{base_url}\n\n", "dim"),
                ("What you can do:\n", "bold"),
                ("  💬  Chat naturally — ask questions, get answers\n", ""),
                ("  📁  File ops — read/write/search files in current dir\n", ""),
                ("  💻  Terminal — run shell commands\n", ""),
                ("  🔍  Web search — fetch live information\n", ""),
                ("  🤖  Multi-agent — delegate tasks to sub-agents\n", ""),
                ("  💾  Sessions — save/resume conversations\n", ""),
                (f"\n{tool_count} tools ready  ·  {session_count} saved sessions  ·  ", "dim"),
                ("/help", "bold cyan"),
                (" for more", "dim"),
            ),
            title=Text("🤖 Kairos", style=self.skin["agent_color"]),
            border_style=self.skin["agent_color"],
            box=self.skin["box"],
            padding=(1, 2),
        )
        self.console.print(panel)
        self.console.print()

    def tool_hint(self, name: str, message: str = "") -> None:
        """Show a subtle one-line tool activity indicator. Always visible."""
        if message:
            self.console.print(f"  [dim]🔧 {name}[/] [dim]{message}[/]")
        else:
            self.console.print(f"  [dim]🔧 {name}...[/]")

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
            ("/retry", "Resend the last message (after timeout/error)"),
            ("/undo", "Undo the last exchange (user message + agent reply)"),
            ("/background <prompt>", "Run a prompt in the background (non-blocking)"),
            ("/goal [status|pause|resume|clear|<text>]", "Set or manage a persistent goal across turns"),
            ("/reasoning [on|off]", "Toggle display of model reasoning/thinking"),
            ("/yolo", "Toggle YOLO mode — bypass dangerous command checks"),
            ("/edit", "Open multi-line editor for code/long text input"),
            ("/model <name>", "Switch model (e.g., /model gpt-4)"),
            ("/skin <name>", "Switch skin (default/hacker/retro/minimal)"),
            ("/tools", "List available tools"),
            ("/verbose", "Toggle verbose tool output"),
            ("/cron list", "List scheduled cron jobs"),
            ("/run <query>", "Run a one-shot query"),
            ("/save <name>", "Save current session"),
            ("/sessions", "List saved sessions"),
            ("/session rename <old> <new>", "Rename a saved session"),
            ("/session delete <name>", "Delete a saved session"),
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

        tool_dict = tools or get_all_tools()
        tool_list = [(name, info.get("schema", {}).get("function", {})) for name, info in tool_dict.items()]

        if not tool_list:
            self.info("No tools registered.")
            return

        table = Table(
            title=f"Available Tools ({len(tool_list)})",
            border_style=self.skin["info_color"],
            box=self.skin["box"],
        )
        table.add_column("Tool", style="bold cyan")
        table.add_column("Description", style="dim")
        table.add_column("Params", style="dim")

        for name, func in tool_list:
            desc = (func.get("description", "") or "")[:60]
            params = ", ".join(
                func.get("parameters", {}).get("properties", {}).keys()
            )[:50]
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
        """Prompt for user input with Rich styling.
        
        When in YOLO mode, the prompt includes a ⚡ indicator.
        Multi-line paste is auto-detected: if the pasted text contains
        newlines, it's captured as-is.
        """
        style = self.skin['user_color']
        if self._yolo_mode:
            style = "bright_yellow"
            prompt_text = f"⚡ {prompt_text}"
        return Prompt.ask(f"[{style}]{prompt_text}[/]")

    def multiline_prompt(self, prompt_text: str = "edit") -> str:
        """Capture multi-line input. 
        
        Prints instructions, then reads lines until '.' on its own line.
        Supports Ctrl+D to finish. Returns joined content.
        """
        info_color = self.skin.get("info_color", "blue")
        self.console.print(
            f"[{info_color}]📝 Multi-line mode. Type '.' on its own line to finish, Ctrl+D to cancel.[/]"
        )
        lines: list[str] = []
        try:
            while True:
                line_num = len(lines) + 1
                line = input(f"[{self.skin['user_color']}]{prompt_text}:{line_num:03d}[/] ")
                if line.strip() == ".":
                    break
                lines.append(line)
        except (EOFError, KeyboardInterrupt):
            if not lines:
                return ""
            # User pressed Ctrl+D — accept what we have
        return "\n".join(lines)

    @property
    def yolo_mode(self) -> bool:
        return self._yolo_mode

    def toggle_yolo(self) -> bool:
        """Toggle YOLO mode. Returns new state."""
        self._yolo_mode = not self._yolo_mode
        return self._yolo_mode

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
