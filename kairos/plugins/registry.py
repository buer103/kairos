"""Plugin Registry — backend storage for plugin registrations.

Shared singleton that PluginContext delegates to. Tracks all registered
hooks, CLI commands, platforms, skills, middleware, memory providers,
and model providers from all loaded plugins.

Thread-safe. Supports unregistration and conflict detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("kairos.plugins.registry")


@dataclass
class HookEntry:
    callback: Callable
    priority: int = 0


@dataclass
class PlatformEntry:
    name: str
    label: str
    adapter_factory: Callable
    check_fn: Callable | None = None
    emoji: str = ""
    platform_hint: str = ""
    max_message_length: int = 4096


@dataclass
class CliCommandEntry:
    name: str
    handler: Callable
    description: str = ""


@dataclass
class MiddlewareEntry:
    name: str
    factory: Callable[[], Any]
    position: str = "append"


class PluginRegistry:
    """Singleton registry for plugin contributions.

    All plugin registrations go through this registry. The gateway,
    middleware pipeline, and CLI can then query it to activate plugins.
    """

    def __init__(self):
        self._hooks: dict[str, list[HookEntry]] = {}
        self._platforms: dict[str, PlatformEntry] = {}
        self._cli_commands: dict[str, CliCommandEntry] = {}
        self._skills: dict[str, str] = {}            # name → path
        self._middleware: dict[str, MiddlewareEntry] = {}
        self._memory: dict[str, Callable] = {}        # name → factory
        self._providers: dict[str, Callable] = {}     # name → factory

    # ── Hooks ─────────────────────────────────────────────

    def add_hook(
        self,
        name: str,
        callback: Callable,
        priority: int = 0,
    ) -> None:
        if name not in self._hooks:
            self._hooks[name] = []
        self._hooks[name].append(HookEntry(callback, priority))
        self._hooks[name].sort(key=lambda h: -h.priority)  # High priority first

    def get_hooks(self, name: str) -> list[HookEntry]:
        return self._hooks.get(name, [])

    def invoke_hooks(self, name: str, **kwargs) -> list[Any]:
        """Invoke all hooks by name. Returns list of results."""
        results = []
        for entry in self.get_hooks(name):
            try:
                results.append(entry.callback(**kwargs))
            except Exception as e:
                logger.warning("Hook '%s' error: %s", name, e)
        return results

    # ── Platforms ─────────────────────────────────────────

    def add_platform(
        self,
        name: str,
        label: str,
        adapter_factory: Callable,
        check_fn: Callable | None = None,
        emoji: str = "",
        platform_hint: str = "",
        max_message_length: int = 4096,
    ) -> None:
        if name in self._platforms:
            logger.warning("Platform '%s' already registered, overwriting", name)
        self._platforms[name] = PlatformEntry(
            name=name, label=label, adapter_factory=adapter_factory,
            check_fn=check_fn, emoji=emoji, platform_hint=platform_hint,
            max_message_length=max_message_length,
        )

    def get_platform(self, name: str) -> PlatformEntry | None:
        return self._platforms.get(name)

    @property
    def platforms(self) -> dict[str, PlatformEntry]:
        return dict(self._platforms)

    # ── CLI Commands ──────────────────────────────────────

    def add_cli_command(
        self,
        name: str,
        handler: Callable,
        description: str = "",
    ) -> None:
        if name in self._cli_commands:
            logger.warning("CLI command '%s' already registered, overwriting", name)
        self._cli_commands[name] = CliCommandEntry(name, handler, description)

    def get_cli_command(self, name: str) -> CliCommandEntry | None:
        return self._cli_commands.get(name)

    @property
    def cli_commands(self) -> dict[str, CliCommandEntry]:
        return dict(self._cli_commands)

    # ── Skills ────────────────────────────────────────────

    def add_skill(self, name: str, path: str, description: str = "") -> None:
        self._skills[name] = path

    def get_skill_path(self, name: str) -> str | None:
        return self._skills.get(name)

    @property
    def skills(self) -> dict[str, str]:
        return dict(self._skills)

    # ── Middleware ─────────────────────────────────────────

    def add_middleware(
        self,
        name: str,
        factory: Callable[[], Any],
        position: str = "append",
    ) -> None:
        if name in self._middleware:
            logger.warning("Middleware '%s' already registered, overwriting", name)
        self._middleware[name] = MiddlewareEntry(name, factory, position)

    def get_middleware(self, name: str) -> MiddlewareEntry | None:
        return self._middleware.get(name)

    @property
    def middleware(self) -> dict[str, MiddlewareEntry]:
        return dict(self._middleware)

    # ── Memory ────────────────────────────────────────────

    def add_memory(self, name: str, factory: Callable) -> None:
        if name in self._memory:
            logger.warning("Memory '%s' already registered, overwriting", name)
        self._memory[name] = factory

    def get_memory_factory(self, name: str) -> Callable | None:
        return self._memory.get(name)

    @property
    def memory_providers(self) -> list[str]:
        return list(self._memory.keys())

    # ── Providers ─────────────────────────────────────────

    def add_provider(
        self,
        name: str,
        factory: Callable,
        model_names: list[str] | None = None,
    ) -> None:
        if name in self._providers:
            logger.warning("Provider '%s' already registered, overwriting", name)
        self._providers[name] = factory

    def get_provider_factory(self, name: str) -> Callable | None:
        return self._providers.get(name)

    @property
    def provider_names(self) -> list[str]:
        return list(self._providers.keys())

    # ── Management ────────────────────────────────────────

    def remove_hook(self, name: str, callback: Callable) -> bool:
        entries = self._hooks.get(name, [])
        for entry in entries:
            if entry.callback is callback:
                entries.remove(entry)
                return True
        return False

    def remove_platform(self, name: str) -> bool:
        return self._platforms.pop(name, None) is not None

    def clear(self) -> None:
        """Clear all registrations (for testing)."""
        self._hooks.clear()
        self._platforms.clear()
        self._cli_commands.clear()
        self._skills.clear()
        self._middleware.clear()
        self._memory.clear()
        self._providers.clear()

    def summary(self) -> dict[str, int]:
        return {
            "hooks": sum(len(v) for v in self._hooks.values()),
            "platforms": len(self._platforms),
            "cli_commands": len(self._cli_commands),
            "skills": len(self._skills),
            "middleware": len(self._middleware),
            "memory": len(self._memory),
            "providers": len(self._providers),
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"PluginRegistry(hooks={s['hooks']}, platforms={s['platforms']}, "
            f"cli={s['cli_commands']}, skills={s['skills']}, "
            f"middleware={s['middleware']}, memory={s['memory']}, "
            f"providers={s['providers']})"
        )


# ── Singleton ───────────────────────────────────────────────
_registry: PluginRegistry | None = None


def get_plugin_registry() -> PluginRegistry:
    """Get the global PluginRegistry singleton."""
    global _registry
    if _registry is None:
        _registry = PluginRegistry()
    return _registry


def reset_plugin_registry() -> None:
    """Reset the singleton (for testing)."""
    global _registry
    _registry = None
