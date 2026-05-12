"""Security middleware — integrates file safety, path security, content
redaction, guardrails, and interactive permission control into the Kairos
middleware pipeline.

Lifecycle hooks used:
  - before_model: validate user input (prompt injection, length, binary)
  - after_model:  validate LLM output (key leaks, PII)
  - wrap_tool_call: validate tool args, check permissions (BLOCK/ASK/TRUST)
  - after_tool:    validate tool result before returning to LLM

Permission model (Claude Code 3-level):
  - BLOCK: always deny (e.g. cronjob)
  - ASK:   prompt user (e.g. terminal, write_file)
  - TRUST: always allow (e.g. read_file, search_files)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from kairos.core.middleware import Middleware
from kairos.core.state import ThreadState
from kairos.security.file_safety import FileSafetyChecker
from kairos.security.path_security import PathSecurity
from kairos.security.url_safety import URLSafety
from kairos.security.content_redact import ContentRedactor
from kairos.security.guardrails import InputGuard, OutputGuard, ToolGuard
from kairos.security.permission import (
    PermissionAction,
    PermissionLevel,
    PermissionManager,
    PermissionRequest,
    ToolPolicy,
)

logger = logging.getLogger("kairos.security")


class SecurityMiddleware(Middleware):
    """Comprehensive security layer for the agent pipeline.

    Configuration:
        allowed_paths: list[str] — base dirs the agent is allowed to access
        max_input_chars: int — maximum user input length
        redact_output: bool — auto-redact API keys/PII in output
        block_dangerous_files: bool — reject tools targeting *.exe, *.sh, etc.
        block_internal_urls: bool — reject tools fetching localhost/private IPs
        strict_tool_args: bool — validate path/url params in tool arguments
        permission_manager: PermissionManager | None — 3-level interactive
            permission control. If omitted, uses defaults (TRUST read-only,
            ASK write, BLOCK cron).
        permission_timeout: float — max seconds to wait for user response

    Usage:
        from kairos.security.permission import PermissionManager
        pm = PermissionManager()
        agent = Agent(middlewares=[..., SecurityMiddleware(permission_manager=pm)])
    """

    def __init__(
        self,
        allowed_paths: list[str] | None = None,
        max_input_chars: int = 50000,
        redact_output: bool = True,
        block_dangerous_files: bool = True,
        block_internal_urls: bool = True,
        strict_tool_args: bool = True,
        permission_manager: PermissionManager | None = None,
        permission_timeout: float = 120.0,
    ):
        self._input_guard = InputGuard(max_length=max_input_chars)
        self._output_guard = OutputGuard()
        self._tool_guard = ToolGuard()
        self._content_redactor = ContentRedactor() if redact_output else None
        self._path_security = PathSecurity(allowed_roots=allowed_paths or [])
        self._url_safety = URLSafety()
        self._file_safety = FileSafetyChecker()
        self._block_dangerous = block_dangerous_files
        self._block_internal = block_internal_urls
        self._strict = strict_tool_args
        self._perm = permission_manager or PermissionManager()
        self._perm_timeout = permission_timeout

    @property
    def permission_manager(self) -> PermissionManager:
        return self._perm

    # ---- Input validation (before_model) --------------------------------

    def before_model(self, state: ThreadState, runtime: dict[str, Any]) -> None:
        """Validate the user message before sending to the model."""
        user_message = runtime.get("user_message", "")
        if not user_message:
            return

        ok, reason = self._input_guard.validate_input(user_message)
        if not ok:
            logger.warning("Input blocked: %s", reason[:200])
            raise SecurityViolation(f"Input rejected: {reason}")

    # ---- Output validation (after_model) --------------------------------

    def after_model(self, state: ThreadState, runtime: dict[str, Any]) -> None:
        """Validate and sanitize model output."""
        messages = state.messages
        if not messages:
            return

        last = messages[-1]
        if last.get("role") != "assistant":
            return

        content = last.get("content", "")
        if not content:
            return

        ok, reason = self._output_guard.validate_output(content)
        if not ok:
            logger.warning("Output blocked: %s", reason[:200])
            if self._content_redactor:
                last["content"] = self._content_redactor.redact(content)
                logger.info("Output redacted")

    # ---- Tool argument validation + permission check (wrap_tool_call) ----

    def wrap_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        executor,
        **kwargs,
    ) -> Any:
        """Validate tool arguments, check permissions, execute, validate result."""
        # Security guardrails first
        if self._strict:
            ok, reason = self._tool_guard.validate_tool_args(tool_name, tool_args)
            if not ok:
                logger.warning("Tool args blocked: %s — %s", tool_name, reason)
                return {"error": reason, "kind": "security_violation"}

            self._validate_tool_args_deep(tool_name, tool_args)

        # Permission check (BLOCK/ASK/TRUST)
        if not self._check_permission_sync(tool_name, tool_args):
            return {"error": "Permission denied by user", "kind": "permission_denied"}

        result = executor(tool_name, tool_args, **kwargs)

        if self._strict:
            ok, reason = self._tool_guard.validate_tool_result(tool_name, result)
            if not ok:
                logger.warning("Tool result blocked: %s — %s", tool_name, reason)
                if self._content_redactor and isinstance(result, dict):
                    result = {"output": self._content_redactor.redact_dict(result)}
                    logger.info("Tool result redacted: %s", tool_name)

        return result

    # ---- Permission check (sync, for middleware compatibility) --------

    def _check_permission_sync(self, tool_name: str, args: dict[str, Any]) -> bool:
        """Check permission synchronously.

        Uses asyncio.run() to bridge async PermissionManager into sync middleware.
        """
        session_id = getattr(self, "_session_id", "default")
        path = args.get("path") or args.get("file_path") or args.get("command") or ""

        request = PermissionRequest(
            session_id=session_id,
            tool_name=tool_name,
            description=_describe_tool_call(tool_name, args),
            params=args,
            path=str(path)[:200],
        )

        # Fast path: TRUST/BLOCK → no async needed
        level = self._perm.get_effective_level(tool_name, session_id)
        if level == PermissionLevel.TRUST or self._perm._auto_approve:
            return True
        if level == PermissionLevel.BLOCK:
            return False

        # Check session grants
        session_key = (tool_name, str(path))
        if session_id in self._perm._session_grants:
            if session_key in self._perm._session_grants[session_id]:
                return True

        # ASK: need interactive prompt
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            return loop.run_until_complete(
                asyncio.wait_for(
                    self._perm.request(request, timeout=self._perm_timeout),
                    timeout=self._perm_timeout,
                )
            )
        except asyncio.TimeoutError:
            logger.warning("Permission prompt timeout for %s", tool_name)
            return False

    # ---- Deep validation helpers -----------------------------------------

    def _validate_tool_args_deep(self, tool_name: str, args: dict[str, Any]) -> None:
        """Additional path/URL checks beyond basic guardrails."""
        for key in ("path", "file_path", "directory", "workdir"):
            if key in args and isinstance(args[key], str):
                p = args[key]
                if self._block_dangerous:
                    ok, reason = self._file_safety.is_safe(p, "path")
                    if not ok and "extension is blocked" in reason:
                        raise SecurityViolation(f"Dangerous file extension: {p}")
                if not self._path_security.is_path_allowed(p):
                    raise SecurityViolation(f"Path not allowed: {p}")

        for key in ("url", "link", "endpoint", "api_url", "source"):
            if key in args and isinstance(args[key], str):
                url = args[key]
                if self._block_internal:
                    ok, reason = self._url_safety.check_url(url)
                    if not ok:
                        raise SecurityViolation(f"URL blocked: {url} — {reason}")


class SecurityViolation(Exception):
    """Raised when a security policy is violated."""
    pass


def _describe_tool_call(tool_name: str, args: dict[str, Any]) -> str:
    """Generate human-readable description of a tool call for permission prompts."""
    if tool_name == "write_file":
        return f"Write file: {args.get('path', '?')}"
    if tool_name == "patch":
        return f"Edit file: {args.get('path', '?')}"
    if tool_name == "terminal":
        cmd = args.get("command", "")[:120]
        return f"Shell: {cmd}"
    if tool_name == "delegate_task":
        goal = str(args.get("goal", ""))[:80]
        return f"Delegate task: {goal}"
    if tool_name == "cronjob":
        return f"Cron mutation: {tool_name}"
    return f"Execute: {tool_name}"
