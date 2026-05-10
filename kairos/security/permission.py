"""Permission system — 3-level interactive tool execution control.

Levels (Claude Code model):
  - BLOCK:   Always deny (never ask). For known-dangerous tools.
  - ASK:     Prompt user for approval. Default for powerful tools.
  - TRUST:   Always allow (never ask). For read-only / sandboxed tools.

Features:
  - ToolPolicy: per-tool rules (BLOCK/ASK/TRUST) with path/wildcard scoping
  - PermissionManager: async service, request/response, session-level remember
  - Session-scoped grants: "Always allow for this session" per tool+path
  - Auto-approve mode: for non-interactive / CI sessions
  - CLI integration: Textual-free, Rich-based prompt via PermissionManager callback
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine

logger = logging.getLogger("kairos.permission")


class PermissionLevel(str, Enum):
    """Three-level permission model."""

    BLOCK = "block"  # Always deny — never ask
    ASK = "ask"      # Prompt user for approval
    TRUST = "trust"  # Always allow — never ask


class PermissionAction(str, Enum):
    """User response to a permission request."""

    ALLOW_ONCE = "allow_once"
    ALLOW_SESSION = "allow_session"
    DENY = "deny"


@dataclass
class ToolPolicy:
    """Per-tool permission policy.

    Wildcards supported: 'terminal', 'write_file', 'web_*' etc.
    """

    tool_pattern: str  # e.g. "write_file", "terminal", "web_*"
    level: PermissionLevel = PermissionLevel.ASK
    path_patterns: list[str] = field(default_factory=list)  # paths this policy applies to
    reason: str = ""

    def matches(self, tool_name: str) -> bool:
        return fnmatch.fnmatch(tool_name, self.tool_pattern)


@dataclass
class PermissionRequest:
    """A permission request from a tool about to execute."""

    id: str = field(default_factory=lambda: f"perm_{uuid.uuid4().hex[:12]}")
    session_id: str = ""
    tool_name: str = ""
    description: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    path: str = ""  # affected path, if any

    def summary(self) -> str:
        """One-line summary for display."""
        parts = [f"[{self.tool_name}]"]
        if self.path:
            parts.append(self.path)
        if self.description:
            parts.append(self.description[:80])
        return " ".join(parts)


@dataclass
class PermissionResponse:
    """Response to a permission request."""

    request_id: str
    action: PermissionAction
    grant_scoped: bool = False  # if True, remember for session


class PermissionManager:
    """Async permission service for tool execution control.

    Usage:
        pm = PermissionManager()
        pm.set_prompt_callback(my_async_prompt_fn)  # for CLI integration

        # In middleware:
        ok = await pm.request(PermissionRequest(
            session_id="s1", tool_name="terminal",
            description="rm -rf /tmp/old",
        ))
        if not ok:
            return {"error": "Permission denied"}
    """

    def __init__(self, auto_approve: bool = False):
        self._policies: dict[str, ToolPolicy] = {}
        self._pending: dict[str, asyncio.Queue[PermissionAction]] = {}
        self._session_grants: dict[str, set[tuple[str, str]]] = {}  # session_id -> {(tool, path), ...}
        self._session_level: dict[str, PermissionLevel] = {}  # per-session override
        self._auto_approve = auto_approve
        self._prompt_callback: Callable[
            [PermissionRequest], Coroutine[Any, Any, PermissionAction | None]
        ] | None = None
        self._default_policy = ToolPolicy("*", level=PermissionLevel.ASK)

        # Register default policies
        self._register_defaults()

    # ── Policy Management ──────────────────────────────────────────

    def set_policy(self, policy: ToolPolicy) -> None:
        """Register or update a tool policy."""
        self._policies[policy.tool_pattern] = policy

    def set_default_level(self, level: PermissionLevel) -> None:
        """Change the default policy for unmatched tools."""
        self._default_policy.level = level

    def set_session_level(self, session_id: str, level: PermissionLevel) -> None:
        """Override all policies for a specific session."""
        self._session_level[session_id] = level

    def get_effective_level(self, tool_name: str, session_id: str) -> PermissionLevel:
        """Resolve the effective permission level for a tool+session."""
        if session_id in self._session_level:
            return self._session_level[session_id]
        for policy in self._policies.values():
            if policy.matches(tool_name):
                return policy.level
        return self._default_policy.level

    def set_prompt_callback(
        self,
        cb: Callable[[PermissionRequest], Coroutine[Any, Any, PermissionAction | None]],
    ) -> None:
        """Set async callback for interactive permission prompts.

        The callback receives a PermissionRequest and should return:
          - PermissionAction.ALLOW_ONCE / ALLOW_SESSION / DENY
          - None (timeout → treat as DENY)
        """
        self._prompt_callback = cb

    def set_auto_approve(self, enabled: bool) -> None:
        """Enable/disable auto-approve mode for all tools."""
        self._auto_approve = enabled

    # ── Permission Requests ────────────────────────────────────────

    async def request(self, request: PermissionRequest, timeout: float = 120.0) -> bool:
        """Request permission to execute a tool.

        Returns True if allowed, False if denied.
        """
        level = self.get_effective_level(request.tool_name, request.session_id)

        # BLOCK: always deny
        if level == PermissionLevel.BLOCK:
            logger.info("Permission BLOCKED: %s", request.summary())
            return False

        # TRUST or auto-approve: always allow
        if level == PermissionLevel.TRUST or self._auto_approve:
            return True

        # Check session-scoped grants
        session_key = (request.tool_name, request.path)
        if request.session_id in self._session_grants:
            if session_key in self._session_grants[request.session_id]:
                return True

        # ASK: prompt user
        action = PermissionAction.DENY
        if self._prompt_callback:
            try:
                result = await asyncio.wait_for(
                    self._prompt_callback(request),
                    timeout=timeout,
                )
                if result is not None:
                    action = result
            except asyncio.TimeoutError:
                logger.warning("Permission prompt timeout for %s", request.summary())

        if action == PermissionAction.DENY:
            logger.info("Permission DENIED: %s", request.summary())
            return False

        if action == PermissionAction.ALLOW_SESSION:
            self._grant_session(request)

        logger.info("Permission GRANTED (%s): %s", action.value, request.summary())
        return True

    # ── Session Grants ─────────────────────────────────────────────

    def _grant_session(self, request: PermissionRequest) -> None:
        """Remember this tool+path combo for the session."""
        if request.session_id not in self._session_grants:
            self._session_grants[request.session_id] = set()
        self._session_grants[request.session_id].add((request.tool_name, request.path))

    def clear_session_grants(self, session_id: str) -> None:
        """Clear all session-level grants."""
        self._session_grants.pop(session_id, None)
        self._session_level.pop(session_id, None)

    # ── Default Policies ───────────────────────────────────────────

    def _register_defaults(self) -> None:
        """Register sensible default tool policies."""
        # Read-only tools → TRUST
        for read_tool in ("read_file", "search_files", "skill_view", "skills_list",
                          "session_search", "memory", "list_providers", "list_files"):
            self.set_policy(ToolPolicy(read_tool, level=PermissionLevel.TRUST,
                                       reason="Read-only, no side effects"))

        # Destructive tools → ASK
        for write_tool in ("write_file", "patch", "terminal", "delegate_task"):
            self.set_policy(ToolPolicy(write_tool, level=PermissionLevel.ASK,
                                       reason="Can modify filesystem or spawn processes"))

        # Explicitly dangerous → BLOCK
        self.set_policy(ToolPolicy("cronjob", level=PermissionLevel.BLOCK,
                                   reason="Cron mutation requires explicit opt-in"))
