"""Plugin Context — Hermes-compatible unified registration interface.

Provides a single facade (PluginContext) through which plugins register:
  - tools (register_tool)
  - hooks (register_hook)
  - CLI commands (register_cli_command)
  - platform adapters (register_platform)
  - skills (register_skill)
  - middleware (register_middleware)
  - memory providers (register_memory)
  - provider adapters (register_provider)

The PluginContext bridges to Kairos' internal registries (tool registry,
middleware pipeline, gateway manager, etc.) with validation and conflict
detection.

Hermes equivalent: hermes_cli/plugins.py PluginContext
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger("kairos.plugins.context")


# ============================================================================
# PluginContext
# ============================================================================


class PluginContext:
    """Unified registration interface for Kairos plugins.

    A single `register(ctx)` function in each plugin receives a PluginContext
    and calls its methods to register capabilities. This is the Hermes model.

    Usage:
        def register(ctx: PluginContext):
            ctx.register_tool("my_tool", toolset="custom", schema=MySchema, handler=my_handler)
            ctx.register_hook("before_model", my_callback)
            ctx.register_platform("irc", "IRC", irc_factory, check_fn=check_irc, emoji="💬")
    """

    def __init__(self, plugin_name: str = "unknown"):
        self._plugin_name = plugin_name
        self._registrations: list[str] = []  # Track what was registered

    # ── Tool registration ────────────────────────────────────

    def register_tool(
        self,
        name: str,
        toolset: str = "plugin",
        schema: dict[str, Any] | None = None,
        handler: Callable | None = None,
        description: str = "",
        requires: list[str] | None = None,
    ) -> None:
        """Register a tool.

        Args:
            name: Tool name (must be unique across registry)
            toolset: Category/toolset name for grouping
            schema: OpenAI-compatible function schema dict
            handler: Async or sync callable that executes the tool
            description: Human-readable description
            requires: List of pip packages or system deps
        """
        from kairos.tools.registry import register_plugin_tool

        if schema is None:
            schema = {
                "name": name,
                "description": description or f"Plugin tool: {name}",
                "parameters": {"type": "object", "properties": {}},
            }

        register_plugin_tool(
            name=name,
            handler=handler or (lambda **kw: "not implemented"),
            schema={
                "type": "function",
                "function": {
                    "name": name,
                    "description": description or f"Plugin tool: {name}",
                    "parameters": schema.get("parameters", {"type": "object", "properties": {}}),
                },
            },
            category=toolset or "plugin",
        )
        self._registrations.append(f"tool:{name}")
        logger.debug("Plugin '%s' registered tool: %s", self._plugin_name, name)

    # ── Hook registration ───────────────────────────────────

    def register_hook(
        self,
        hook_name: str,
        callback: Callable,
        priority: int = 0,
    ) -> None:
        """Register a lifecycle hook.

        Valid hook names:
            before_agent, after_agent, before_model, after_model,
            before_tool, after_tool, on_error, on_session_start,
            on_session_end, pre_gateway_dispatch
        """
        from kairos.plugins.registry import get_plugin_registry
        reg = get_plugin_registry()
        reg.add_hook(hook_name, callback, priority=priority)
        self._registrations.append(f"hook:{hook_name}")
        logger.debug("Plugin '%s' registered hook: %s", self._plugin_name, hook_name)

    # ── CLI command registration ─────────────────────────────

    def register_cli_command(
        self,
        name: str,
        handler: Callable,
        description: str = "",
    ) -> None:
        """Register a CLI subcommand (e.g. `kairos mycmd`).

        Args:
            name: CLI command name (e.g. "deploy")
            handler: Callable(args: list[str]) that handles the command
            description: Help text shown in `kairos --help`
        """
        from kairos.plugins.registry import get_plugin_registry
        reg = get_plugin_registry()
        reg.add_cli_command(name, handler, description=description)
        self._registrations.append(f"cli_cmd:{name}")
        logger.debug("Plugin '%s' registered CLI command: %s", self._plugin_name, name)

    # ── Platform registration ────────────────────────────────

    def register_platform(
        self,
        name: str,
        label: str,
        adapter_factory: Callable,
        check_fn: Callable | None = None,
        emoji: str = "",
        platform_hint: str = "",
        max_message_length: int = 4096,
    ) -> None:
        """Register a gateway platform adapter.

        Args:
            name: Platform identifier (e.g. "irc", "teams")
            label: Human-readable label
            adapter_factory: Callable(config) → BasePlatformAdapter
            check_fn: Optional callable → bool (returns True if platform available)
            emoji: Emoji icon for the platform
            platform_hint: Hint injected into system prompt for this platform
            max_message_length: Max chars per message for this platform
        """
        from kairos.plugins.registry import get_plugin_registry
        reg = get_plugin_registry()
        reg.add_platform(
            name=name,
            label=label,
            adapter_factory=adapter_factory,
            check_fn=check_fn,
            emoji=emoji,
            platform_hint=platform_hint,
            max_message_length=max_message_length,
        )
        self._registrations.append(f"platform:{name}")
        logger.debug("Plugin '%s' registered platform: %s", self._plugin_name, name)

    # ── Skill registration ──────────────────────────────────

    def register_skill(
        self,
        name: str,
        path: str,
        description: str = "",
    ) -> None:
        """Register a read-only skill (shipped with the plugin).

        Args:
            name: Skill name
            path: Absolute path to the skill directory (contains SKILL.md)
            description: Short description
        """
        from kairos.plugins.registry import get_plugin_registry
        reg = get_plugin_registry()
        reg.add_skill(name, path, description=description)
        self._registrations.append(f"skill:{name}")
        logger.debug("Plugin '%s' registered skill: %s", self._plugin_name, name)

    # ── Middleware registration ──────────────────────────────

    def register_middleware(
        self,
        name: str,
        factory: Callable[[], Any],
        position: str = "append",  # "prepend" | "append" | "before:<name>" | "after:<name>"
    ) -> None:
        """Register a middleware layer.

        Args:
            name: Unique middleware name
            factory: Callable() → Middleware instance
            position: Where to insert in the pipeline
        """
        from kairos.plugins.registry import get_plugin_registry
        reg = get_plugin_registry()
        reg.add_middleware(name, factory, position=position)
        self._registrations.append(f"middleware:{name}")
        logger.debug("Plugin '%s' registered middleware: %s", self._plugin_name, name)

    # ── Memory provider registration ─────────────────────────

    def register_memory(
        self,
        name: str,
        backend_factory: Callable,
    ) -> None:
        """Register a memory backend provider.

        Args:
            name: Provider name (e.g. "redis", "sqlite")
            backend_factory: Callable() → MemoryBackend
        """
        from kairos.plugins.registry import get_plugin_registry
        reg = get_plugin_registry()
        reg.add_memory(name, backend_factory)
        self._registrations.append(f"memory:{name}")
        logger.debug("Plugin '%s' registered memory: %s", self._plugin_name, name)

    # ── Provider registration ────────────────────────────────

    def register_provider(
        self,
        name: str,
        adapter_factory: Callable,
        model_names: list[str] | None = None,
    ) -> None:
        """Register a model provider adapter.

        Args:
            name: Provider name (e.g. "anthropic", "gemini")
            adapter_factory: Callable(config) → ModelProvider
            model_names: List of model names this provider supports
        """
        from kairos.plugins.registry import get_plugin_registry
        reg = get_plugin_registry()
        reg.add_provider(name, adapter_factory, model_names=model_names or [])
        self._registrations.append(f"provider:{name}")
        logger.debug("Plugin '%s' registered provider: %s", self._plugin_name, name)

    # ── Properties ──────────────────────────────────────────

    @property
    def plugin_name(self) -> str:
        return self._plugin_name

    @property
    def registrations(self) -> list[str]:
        """List of all registrations made through this context."""
        return list(self._registrations)

    def __repr__(self) -> str:
        return (
            f"PluginContext(plugin={self._plugin_name}, "
            f"registrations={len(self._registrations)})"
        )
