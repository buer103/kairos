"""Sandbox middleware — route tool executions through isolated sandboxes.

Picks up sandbox config and wraps tool calls (terminal, execute) to run
in the configured sandbox provider (local / docker / ssh / cloud).

If no sandbox is configured on the agent, execution is passthrough
(identical to current behavior — subprocess.run on the host).
"""

from __future__ import annotations

from typing import Any

from kairos.core.middleware import Middleware
from kairos.core.state import ThreadState
from kairos.sandbox.providers import (
    Sandbox,
    SandboxConfig,
    SandboxProvider,
    create_sandbox,
)

# Tools that should be routed through the sandbox
_SANDBOX_TOOLS = {"terminal", "execute", "bash", "shell", "run_command"}


class SandboxMiddleware(Middleware):
    """Middleware that routes tool execution through the configured sandbox.

    Usage::

        from kairos.sandbox.providers import SandboxConfig, SandboxProvider

        config = SandboxConfig(
            provider=SandboxProvider.DOCKER,
            docker_image="python:3.11-slim",
            timeout=300,
        )
        layers = [SandboxMiddleware(sandbox_config=config), ...]
    """

    def __init__(
        self,
        sandbox_config: SandboxConfig | None = None,
        sandbox: Sandbox | None = None,
    ):
        """
        Args:
            sandbox_config: Create a sandbox from config.
            sandbox: Use an existing sandbox instance (overrides config).
        """
        self._config = sandbox_config
        self._sandbox = sandbox
        if self._sandbox is None and self._config is not None:
            self._sandbox = create_sandbox(self._config)

    @property
    def sandbox(self) -> Sandbox | None:
        return self._sandbox

    @property
    def is_active(self) -> bool:
        """Whether routing is active (non-local sandbox configured)."""
        return (
            self._sandbox is not None
            and self._config is not None
            and self._config.provider != SandboxProvider.LOCAL
        )

    def wrap_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        original: Any,
        *,
        state: ThreadState | None = None,
        **kwargs,
    ) -> Any:
        """Intercept tool calls and route through sandbox if applicable."""
        # Passthrough if no sandbox or not a sandbox-eligible tool
        if self._sandbox is None or tool_name not in _SANDBOX_TOOLS:
            return original(tool_name, tool_args)

        # Extract command from tool args
        command = tool_args.get("command") or tool_args.get("cmd") or ""
        if not command:
            return original(tool_name, tool_args)

        # Execute through sandbox
        timeout = tool_args.get("timeout")
        result = self._sandbox.execute(command, timeout=timeout)

        # Return in a format compatible with tool result expectations
        return {
            "stdout": result.output,
            "stderr": "",
            "exit_code": result.exit_code,
            "duration_ms": result.duration_ms,
            "sandbox_provider": result.provider,
            "sandbox_id": result.sandbox_id,
        }

    def cleanup(self) -> None:
        """Clean up sandbox resources when agent shuts down."""
        if self._sandbox:
            try:
                self._sandbox.cleanup()
            except Exception:
                pass
