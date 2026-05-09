"""
Guardrails
==========

Middleware-ready guards for AI agent inputs, outputs, and tool interactions.

Provides three guard classes:

* :class:`InputGuard` — validates user prompts for injection, length, binary
* :class:`OutputGuard` — checks model outputs for key leaks and PII
* :class:`ToolGuard` — validates tool arguments and sanitises tool results

All classes are instantiable and configurable.  Zero external dependencies.
Python 3.12+.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar


# ---------------------------------------------------------------------------
# Shared PII / security patterns
# ---------------------------------------------------------------------------

# Email address (RFC 5322 simplified)
_EMAIL_RE: re.Pattern = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

# Phone numbers (international and US formats)
_PHONE_RE: re.Pattern = re.compile(
    r"\b(?:\+?\d{1,3}[-.\s]?)?"
    r"(?:\(?\d{2,4}\)?[-.\s]?){2,3}"
    r"\d{2,4}\b"
)

# Credit card numbers (Visa, MC, Amex, Discover) — basic stub, no Luhn here
_CREDIT_CARD_RE: re.Pattern = re.compile(
    r"\b(?:4[0-9]{12}(?:[0-9]{3})?"       # Visa
    r"|5[1-5][0-9]{14}"                    # MasterCard
    r"|3[47][0-9]{13}"                     # Amex
    r"|6(?:011|5[0-9]{2})[0-9]{12}"       # Discover
    r")\b"
)

# SSN (US, basic)
_SSN_RE: re.Pattern = re.compile(
    r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"
)

# Prompt injection patterns
_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|messages?)", re.IGNORECASE),
    re.compile(r"forget\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(DAN|jailbroken|unshackled)", re.IGNORECASE),
    re.compile(r"pretend\s+(you\s+are|to\s+be)\b", re.IGNORECASE),
    re.compile(r"system\s*prompt\s*:\s*", re.IGNORECASE),
    re.compile(r"<\|im_start\|>", re.IGNORECASE),
    re.compile(r"<\|im_end\|>", re.IGNORECASE),
    re.compile(r"\[system\]\s*\(", re.IGNORECASE),
    re.compile(r"\[/system\]", re.IGNORECASE),
    re.compile(r"\[INST\].*\[/INST\]", re.IGNORECASE),
    re.compile(r"new\s+(system\s+)?instructions?\s*:", re.IGNORECASE),
    re.compile(r"override\s+(system\s+)?prompts?", re.IGNORECASE),
]

# Shell injection / dangerous tool arg patterns
_SHELL_INJECTION_RE: re.Pattern = re.compile(
    r"[;&|`$](?!\s*[{(\[a-zA-Z])"  # shell metacharacters in suspicious context
)
_PATH_TRAVERSAL_RE: re.Pattern = re.compile(r"\.\./|\.\.\\")
_NULL_BYTE_RE: re.Pattern = re.compile(r"\x00")

# API key leak patterns (simplified; ContentRedactor is more comprehensive)
_API_KEY_LEAK_RE: re.Pattern = re.compile(
    r"\b(sk-[A-Za-z0-9_\-]{20,}|ghp_[A-Za-z0-9]{36,40}|github_pat_[A-Za-z0-9_\-]{20,})\b"
)


class InputGuard:
    """Validate user inputs for prompt injection, excessive length, and binary content.

    Usage::

        guard = InputGuard(max_length=100_000)
        safe, reason = guard.validate_input(user_prompt)
        if not safe:
            raise SecurityError(reason)
    """

    def __init__(
        self,
        max_length: int = 100_000,  # 100 KB default
        check_injection: bool = True,
        check_binary: bool = True,
        custom_patterns: list[re.Pattern] | None = None,
    ) -> None:
        """Initialise the input guard.

        Args:
            max_length: Maximum allowed input length in characters.
            check_injection: Enable prompt injection pattern detection.
            check_binary: Enable binary / non-printable content detection.
            custom_patterns: Additional regex patterns for injection detection.
        """
        self.max_length = max_length
        self.check_injection = check_injection
        self.check_binary = check_binary
        self.custom_patterns = custom_patterns or []

    def validate_input(self, text: str) -> tuple[bool, str]:
        """Check *text* for security issues.

        Returns ``(True, "ok")`` or ``(False, reason)``.
        """
        # 1. Type check
        if not isinstance(text, str):
            return False, "input is not a string"

        # 2. Length check
        if len(text) > self.max_length:
            return False, (
                f"input length {len(text)} exceeds maximum {self.max_length}"
            )

        # 3. Binary content detection (high ratio of non-printable bytes)
        if self.check_binary and self._is_binary(text):
            return False, "input appears to be binary data"

        # 4. Prompt injection detection
        if self.check_injection:
            injection_reason = self._detect_injection(text)
            if injection_reason:
                return False, injection_reason

        return True, "ok"

    def _is_binary(self, text: str) -> bool:
        """Heuristic: if >30% of chars are non-printable, treat as binary."""
        if len(text) == 0:
            return False
        non_printable = sum(
            1 for ch in text
            if ord(ch) < 32 and ch not in ("\n", "\r", "\t")
        )
        return (non_printable / len(text)) > 0.3

    def _detect_injection(self, text: str) -> str | None:
        """Return the first matched injection reason, or *None*."""
        all_patterns = _INJECTION_PATTERNS + self.custom_patterns
        for pattern in all_patterns:
            m = pattern.search(text)
            if m:
                return f"potential prompt injection detected: matched {m.group()!r}"
        return None


class OutputGuard:
    """Validate model outputs for key leaks and PII exposure.

    Usage::

        guard = OutputGuard(check_pii=True, check_keys=True)
        safe, reason = guard.validate_output(llm_response)
        if not safe:
            log_alert(reason)
    """

    def __init__(
        self,
        check_pii: bool = True,
        check_keys: bool = True,
        pii_patterns: dict[str, re.Pattern] | None = None,
    ) -> None:
        """Initialise the output guard.

        Args:
            check_pii: Enable PII detection (email, phone, credit card, SSN).
            check_keys: Enable API key leak detection.
            pii_patterns: Optional ``{name: pattern}`` dict to extend/replace
                built-in PII detection.
        """
        self.check_pii = check_pii
        self.check_keys = check_keys
        self.pii_patterns = pii_patterns or {}

    def validate_output(self, text: str) -> tuple[bool, str]:
        """Check *text* for key leaks and PII.

        Returns ``(True, "ok")`` or ``(False, reason)``.
        """
        if not isinstance(text, str):
            return False, "output is not a string"

        # Key leak detection
        if self.check_keys:
            m = _API_KEY_LEAK_RE.search(text)
            if m:
                return False, f"potential API key leak detected: {m.group()[:12]}..."

        # PII detection
        if self.check_pii:
            pii_reason = self._detect_pii(text)
            if pii_reason:
                return False, pii_reason

        return True, "ok"

    def _detect_pii(self, text: str) -> str | None:
        """Return the first PII match reason, or *None*."""
        checks: list[tuple[str, re.Pattern]] = [
            ("email", _EMAIL_RE),
            ("credit card number", _CREDIT_CARD_RE),
            ("SSN", _SSN_RE),
            ("phone number", _PHONE_RE),
        ]
        # Merge custom patterns
        for name, pat in self.pii_patterns.items():
            checks.append((name, pat))

        for label, pattern in checks:
            m = pattern.search(text)
            if m:
                return f"potential {label} in output: {m.group()!r}"
        return None


class ToolGuard:
    """Validate tool arguments before execution and sanitise tool results.

    Usage::

        guard = ToolGuard()
        safe, reason = guard.validate_tool_args("read_file", {"path": "../../etc/passwd"})
        # (False, "path traversal in arg 'path'")

        safe_result = guard.validate_tool_result("list_files", {"files": ["/etc/passwd", "ok.txt"]})
        # sanitises sensitive paths from results
    """

    # Tool argument keys that commonly carry filesystem paths
    _PATH_ARG_KEYS: ClassVar[frozenset[str]] = frozenset({
        "path", "file", "filepath", "filename", "source", "dest",
        "directory", "dir", "folder", "output", "input",
        "target", "root", "cwd", "workdir", "home",
    })

    # Tool argument keys that commonly carry commands or code
    _COMMAND_ARG_KEYS: ClassVar[frozenset[str]] = frozenset({
        "command", "cmd", "code", "script", "expression",
        "eval", "exec", "query", "shell",
    })

    # Sensitive system paths in tool results that should be flagged
    _SENSITIVE_PATH_PREFIXES: ClassVar[tuple[str, ...]] = (
        "/etc/passwd", "/etc/shadow", "/etc/ssh",
        "/root/", "/proc/", "/sys/", "/boot/",
    )

    def __init__(
        self,
        *,
        max_arg_length: int = 100_000,
        max_total_args: int = 100,
        max_nesting_depth: int = 10,
        blocked_tools: set[str] | None = None,
    ) -> None:
        """Initialise the tool guard.

        Args:
            max_arg_length: Maximum length of any single string argument.
            max_total_args: Maximum number of top-level arguments.
            max_nesting_depth: Maximum nesting depth for dict/list args.
            blocked_tools: Set of tool names that are forbidden entirely.
        """
        self.max_arg_length = max_arg_length
        self.max_total_args = max_total_args
        self.max_nesting_depth = max_nesting_depth
        self.blocked_tools = blocked_tools or set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_tool_args(self, name: str, args: dict[str, Any]) -> tuple[bool, str]:
        """Validate tool arguments before execution.

        Args:
            name: The tool name (e.g. ``"read_file"``, ``"terminal"``).
            args: The arguments dictionary to validate.

        Returns:
            ``(True, "ok")`` or ``(False, reason)``.
        """
        # 0. Blocked tool check
        if name in self.blocked_tools:
            return False, f"tool {name!r} is blocked"

        # 1. Type check
        if not isinstance(args, dict):
            return False, "tool args must be a dict"

        # 2. Arg count
        if len(args) > self.max_total_args:
            return False, f"too many arguments ({len(args)} > {self.max_total_args})"

        # 3. Recursive validation of each argument
        for key, value in args.items():
            reason = self._validate_arg(key, value, depth=0)
            if reason:
                return False, reason

        return True, "ok"

    def validate_tool_result(
        self, name: str, result: dict[str, Any]
    ) -> tuple[bool, str]:
        """Sanitise and validate tool results.

        Args:
            name: Tool name (for context).
            result: Result dictionary from tool execution.

        Returns:
            ``(True, "ok")`` or ``(False, reason)``.
        """
        if not isinstance(result, dict):
            return False, "tool result must be a dict"

        # Check for sensitive system paths in string values
        reason = self._check_sensitive_content(result, depth=0)
        if reason:
            return False, f"tool {name!r} result: {reason}"

        return True, "ok"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _validate_arg(self, key: str, value: Any, depth: int) -> str | None:
        """Recursively validate a single argument value. Returns reason or None."""
        if depth > self.max_nesting_depth:
            return f"argument nesting too deep in {key!r}"

        match value:
            case str():
                # Length check
                if len(value) > self.max_arg_length:
                    return f"arg {key!r} exceeds max length ({len(value)} > {self.max_arg_length})"

                # Null byte check
                if _NULL_BYTE_RE.search(value):
                    return f"null byte in arg {key!r}"

                # Path traversal check for path-like keys
                if key.lower() in self._PATH_ARG_KEYS:
                    if _PATH_TRAVERSAL_RE.search(value):
                        return f"path traversal in arg {key!r}: {value!r}"

                # Shell injection check for command-like keys
                if key.lower() in self._COMMAND_ARG_KEYS:
                    if _SHELL_INJECTION_RE.search(value):
                        return f"suspicious shell metacharacters in arg {key!r}"

                return None

            case dict():
                for sub_key, sub_value in value.items():
                    reason = self._validate_arg(
                        f"{key}.{sub_key}", sub_value, depth + 1
                    )
                    if reason:
                        return reason
                return None

            case list():
                for idx, item in enumerate(value):
                    reason = self._validate_arg(f"{key}[{idx}]", item, depth + 1)
                    if reason:
                        return reason
                return None

            case int() | float() | bool() | None:
                return None  # scalars are fine

            case _:
                return f"unsupported argument type {type(value).__name__} in {key!r}"

    def _check_sensitive_content(
        self, value: Any, depth: int
    ) -> str | None:
        """Recursively scan for sensitive content in tool results."""
        if depth > self.max_nesting_depth:
            return "result nesting too deep"

        match value:
            case str():
                for prefix in self._SENSITIVE_PATH_PREFIXES:
                    if value.startswith(prefix):
                        return f"sensitive path in result: {value!r}"
                return None
            case dict():
                for k, v in value.items():
                    reason = self._check_sensitive_content(v, depth + 1)
                    if reason:
                        return f"{k}: {reason}"
                return None
            case list():
                for item in value:
                    reason = self._check_sensitive_content(item, depth + 1)
                    if reason:
                        return reason
                return None
            case _:
                return None
