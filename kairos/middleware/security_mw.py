"""Security middleware — integrates file safety, path security, content
redaction, and guardrails into the Kairos middleware pipeline.

Lifecycle hooks used:
  - before_model: validate user input (prompt injection, length, binary)
  - after_model:  validate LLM output (key leaks, PII)
  - wrap_tool_call: validate tool arguments before execution
  - after_tool:    validate tool result before returning to LLM
"""

from __future__ import annotations

import logging
from typing import Any

from kairos.core.middleware import Middleware
from kairos.core.state import ThreadState
from kairos.security.file_safety import FileSafetyChecker
from kairos.security.path_security import PathSecurity
from kairos.security.url_safety import URLSafety
from kairos.security.content_redact import ContentRedactor
from kairos.security.guardrails import InputGuard, OutputGuard, ToolGuard

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

    Usage:
        from kairos.security.guardrails_middleware import SecurityMiddleware
        agent = Agent(middlewares=[..., SecurityMiddleware(allowed_paths=["/tmp"])])
    """

    def __init__(
        self,
        allowed_paths: list[str] | None = None,
        max_input_chars: int = 50000,
        redact_output: bool = True,
        block_dangerous_files: bool = True,
        block_internal_urls: bool = True,
        strict_tool_args: bool = True,
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

    # ---- Tool argument validation (wrap_tool_call) -----------------------

    def wrap_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        executor,
        **kwargs,
    ) -> Any:
        """Validate tool arguments, execute, validate result."""
        if self._strict:
            ok, reason = self._tool_guard.validate_tool_args(tool_name, tool_args)
            if not ok:
                logger.warning("Tool args blocked: %s — %s", tool_name, reason)
                return {"error": reason, "kind": "security_violation"}

            # Deep path/URL checks
            self._validate_tool_args_deep(tool_name, tool_args)

        result = executor(tool_name, tool_args, **kwargs)

        if self._strict:
            ok, reason = self._tool_guard.validate_tool_result(tool_name, result)
            if not ok:
                logger.warning("Tool result blocked: %s — %s", tool_name, reason)
                if self._content_redactor and isinstance(result, dict):
                    result = {"output": self._content_redactor.redact_dict(result)}
                    logger.info("Tool result redacted: %s", tool_name)

        return result

    # ---- Deep validation helpers -----------------------------------------

    def _validate_tool_args_deep(self, tool_name: str, args: dict[str, Any]) -> None:
        """Additional path/URL checks beyond basic guardrails."""
        # Check path arguments
        for key in ("path", "file_path", "directory", "workdir"):
            if key in args and isinstance(args[key], str):
                p = args[key]
                # Check dangerous files
                if self._block_dangerous:
                    ok, reason = self._file_safety.is_safe(p, "path")
                    if not ok and "extension is blocked" in reason:
                        raise SecurityViolation(f"Dangerous file extension: {p}")

                # Check path traversal
                if not self._path_security.is_path_allowed(p):
                    raise SecurityViolation(f"Path not allowed: {p}")

        # Check URL arguments
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
