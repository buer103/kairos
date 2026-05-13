"""Tab completion engine for Kairos CLI.

Provides:
  - Command completion (chat, run, cron, config, skill, curator)
  - Sub-command completion (cron list|add|pause..., skill install|view|...)
  - Tool name completion (registered tools)
  - Skill name completion (installed skills)
  - File path completion (for skill install)
  - Model name suggestions

Works with readline on Unix and prompt_toolkit when available.
"""

from __future__ import annotations

import os
import shlex
from typing import Any


class Completer:
    """Tab completion engine with context-aware suggestions.

    Usage:
        completer = Completer()
        completer.complete("kairos skil", 6)  # → "skill"
    """

    # Command tree
    COMMANDS = {
        "chat": [],
        "run": [],
        "tui": ["--skin", "--verbose", "--resume", "--model", "--base-url"],
        "cron": ["list", "add", "pause", "resume", "cancel", "remove"],
        "config": ["init", "show"],
        "skill": ["list", "view", "install", "uninstall", "update", "marketplace"],
        "curator": ["status", "clean", "reindex"],
        "gateway": ["start", "stop", "status", "pair"],
        "plugin": ["list", "enable", "disable"],
    }

    GLOBAL_FLAGS = ["--version", "--help", "-h", "--verbose", "-v", "--quiet", "-q"]

    MODEL_NAMES = [
        "gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-4o-mini", "gpt-3.5-turbo",
        "claude-sonnet-4-20250514", "claude-opus-4-20250514",
        "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022",
        "deepseek-chat", "deepseek-reasoner",
        "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash",
    ]

    def __init__(self):
        self._tools_cache: list[str] = []
        self._skills_cache: list[str] = []
        self._models_cache: list[str] = list(self.MODEL_NAMES)
        self._cache_ttl = 0
        self._refresh_cache()

    def _refresh_cache(self) -> None:
        """Refresh tool and skill name caches from registry."""
        import time
        now = time.time()
        if now - self._cache_ttl < 30:
            return  # Cache is fresh

        try:
            from kairos.tools.registry import get_all_tools
            tools = get_all_tools()
            self._tools_cache = sorted(tools.keys())
        except Exception:
            pass

        try:
            from kairos.skills.manager import SkillManager
            mgr = SkillManager()
            mgr.scan()
            self._skills_cache = sorted(e.name for e in mgr.list_skills())
        except Exception:
            pass

        self._cache_ttl = now

    # ── Main completion entry ─────────────────────────────────────────────

    def complete(self, text: str, state: int) -> str | None:
        """Readline completer function. Returns the `state`-th match.

        Compatible with readline.set_completer().
        """
        matches = self.get_matches(text)
        if state < len(matches):
            return matches[state]
        return None

    def get_matches(self, text: str) -> list[str]:
        """Get all possible completions for the given text."""
        self._refresh_cache()

        tokens = self._tokenize(text)
        if not tokens:
            return list(self.COMMANDS.keys()) + self.GLOBAL_FLAGS

        # If text ends with space, we're completing the next argument
        ends_with_space = text.endswith(" ")

        if not ends_with_space and len(tokens) == 1 and not tokens[0].startswith("-"):
            # Completing command name
            return self._match(tokens[0], list(self.COMMANDS.keys()))

        if tokens[0] == "chat":
            return self._complete_chat(tokens[1:], ends_with_space)

        if tokens[0] == "cron":
            return self._complete_cron(tokens[1:], ends_with_space)

        if tokens[0] == "skill":
            return self._complete_skill(tokens[1:], ends_with_space)

        if tokens[0] == "config":
            return self._complete_config(tokens[1:], ends_with_space)

        if tokens[0] == "curator":
            return self._match(tokens[-1] if not ends_with_space else "",
                            self.COMMANDS["curator"])

        if tokens[0] == "gateway":
            return self._match(tokens[-1] if not ends_with_space else "",
                            self.COMMANDS["gateway"])

        if tokens[0] == "plugin":
            return self._complete_plugin(tokens[1:], ends_with_space)

        if tokens[0] in ("--help", "-h", "--version"):
            return []

        # File path completion
        if tokens[0] == "skill" and tokens[1] in ("install",) if len(tokens) > 1 else False:
            path_token = tokens[-1] if not ends_with_space else ""
            return self._complete_path(path_token)

        return []

    # ── Sub-command completions ───────────────────────────────────────────

    def _complete_chat(self, args: list[str], ends_with_space: bool) -> list[str]:
        """Complete chat sub-command args: --skin, --model, --verbose."""
        token = args[-1] if args and not ends_with_space else ""

        if not args:
            return []

        prev = args[-2] if len(args) >= 2 else ""

        if prev == "--skin":
            from kairos.cli.skin_engine import SkinEngine
            engine = SkinEngine()
            skins = [s["name"] for s in engine.list_skins()]
            return self._match(token, skins)

        if prev == "--model":
            return self._match(token, self._models_cache)

        flags = ["--skin", "--model", "--verbose", "--quiet", "--no-stream"]
        remaining = [f for f in flags if f not in args]
        return self._match(token, remaining)

    def _complete_cron(self, args: list[str], ends_with_space: bool) -> list[str]:
        """Complete cron sub-commands."""
        token = args[-1] if args and not ends_with_space else ""
        sub_cmds = self.COMMANDS["cron"]
        return self._match(token, sub_cmds) if len(args) <= 1 else []

    def _complete_skill(self, args: list[str], ends_with_space: bool) -> list[str]:
        """Complete skill sub-commands and skill names."""
        token = args[-1] if args and not ends_with_space else ""

        if not args or ends_with_space:
            return self.COMMANDS["skill"]

        if len(args) == 1:
            return self._match(token, self.COMMANDS["skill"])

        # skill view <name>, skill uninstall <name>, skill update <name>
        if args[0] in ("view", "uninstall", "update") and len(args) == 2:
            return self._match(token, self._skills_cache)

        # skill install <path/url>
        if args[0] == "install" and len(args) == 2:
            path_matches = self._complete_path(token)
            if path_matches:
                return path_matches
            # Suggest common sources
            sources = ["github.com/", "huggingface://", "https://", "~/.kairos/skills/"]
            return self._match(token, sources)

        return []

    def _complete_config(self, args: list[str], ends_with_space: bool) -> list[str]:
        token = args[-1] if args and not ends_with_space else ""
        return self._match(token, self.COMMANDS["config"]) if len(args) <= 1 and not ends_with_space else []

    def _complete_plugin(self, args: list[str], ends_with_space: bool) -> list[str]:
        token = args[-1] if args and not ends_with_space else ""
        return self._match(token, self.COMMANDS["plugin"]) if len(args) <= 1 and not ends_with_space else []

    # ── File path completion ─────────────────────────────────────────────

    def _complete_path(self, token: str) -> list[str]:
        """Complete file/directory paths."""
        token = os.path.expanduser(token)
        dirname = os.path.dirname(token) or "."
        basename = os.path.basename(token)

        if not os.path.isdir(dirname):
            return []

        try:
            entries = os.listdir(dirname)
        except PermissionError:
            return []

        matches = []
        for entry in entries:
            if entry.startswith(basename):
                full = os.path.join(dirname, entry)
                if os.path.isdir(full):
                    matches.append(full + "/")
                else:
                    matches.append(full)
        return sorted(matches)

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Split text into tokens, respecting quotes."""
        import shlex
        try:
            return shlex.split(text)
        except ValueError:
            return text.split()

    @staticmethod
    def _match(prefix: str, candidates: list[str]) -> list[str]:
        """Return candidates that start with prefix."""
        if not prefix:
            return candidates
        lower = prefix.lower()
        return [c for c in candidates if c.lower().startswith(lower)]


# ── Prompt toolkit integration ──────────────────────────────────────────


def setup_readline_completion(completer: Completer | None = None) -> None:
    """Setup readline with tab completion for the Kairos CLI.

    Call once at startup: setup_readline_completion()
    """
    try:
        import readline
        c = completer or Completer()
        readline.set_completer(c.complete)
        readline.parse_and_bind("tab: complete")
        readline.set_completer_delims(" \t\n;")

        # History file
        histfile = os.path.expanduser("~/.kairos/cli_history")
        try:
            readline.read_history_file(histfile)
        except FileNotFoundError:
            pass
        import atexit
        atexit.register(readline.write_history_file, histfile)
    except ImportError:
        pass  # readline not available (Windows, some envs)


def setup_prompt_toolkit_completion() -> Any | None:
    """Return a prompt_toolkit completer if available.

    Usage:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import Completer as PTCompleter
        # ... (user provides when using ptk)
    """
    try:
        from prompt_toolkit.completion import Completer as _PTBase, Completion

        class _KairosPTCompleter(_PTBase):
            def __init__(self):
                self._completer = Completer()

            def get_completions(self, document, complete_event):
                text = document.text_before_cursor
                matches = self._completer.get_matches(text)
                for match in matches:
                    yield Completion(match, start_position=-len(document.get_word_before_cursor()))

        return _KairosPTCompleter()
    except ImportError:
        return None
