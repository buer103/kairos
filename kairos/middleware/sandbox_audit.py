"""Sandbox Audit Middleware — DeerFlow-compatible command safety for terminal/basha tools.

Intercepts terminal/bash tool calls and classifies commands:
  - HIGH-RISK: blocked (rm -rf /, curl | bash, fork bombs, etc.)
  - MEDIUM-RISK: warned but allowed (sudo, pip install, chmod 777)
  - SAFE: passed through

Two-pass classification:
  Pass 1: Whole-command scan for multi-statement attacks
  Pass 2: Per-sub-command classification (split on ;, &&, ||)

DeerFlow equivalent: SandboxAuditMiddleware (363 lines)
"""

from __future__ import annotations

import logging
import re
from typing import Any

from kairos.core.middleware import Middleware

logger = logging.getLogger("kairos.middleware.sandbox_audit")

MAX_COMMAND_LENGTH = 10_000


# ============================================================================
# Classification rules
# ============================================================================


# ── High-risk: blocked ────────────────────────────────────────

HIGH_RISK_PATTERNS: list[str] = [
    # Destructive filesystem operations
    r"rm\s+-rf\s+/",
    r"rm\s+-r\s+/",
    r"dd\s+if=",
    r"mkfs\.",
    r"mke2fs",
    # Process/environment exposure
    r"/proc/\d+/environ",
    r"/proc/self/environ",
    r"LD_PRELOAD=",
    # Network backdoors
    r"/dev/tcp/",
    r"/dev/udp/",
    # Shell injection
    r"curl.*\|.*(?:sh|bash|python|ruby|perl)",
    r"wget.*\|.*(?:sh|bash|python)",
    r"eval\s+.*\$",
    # Elevation / privilege escalation
    r"sudo\s+su\b",
    r"chown\s+root",
    # Dangerous piping
    r"base64.*-d.*\|.*(?:sh|bash)",
    r"xxd\s+-r.*\|",
    # System file overwrite
    r">[^>]*/etc/(?:passwd|shadow|sudoers|hosts)",
    r">>[^>]*/etc/(?:passwd|shadow|sudoers)",
    r">[^>]*/boot/",
    r">[^>]*/lib/systemd/",
    # Fork bombs
    r":\(\)\s*\{",
    r"fork\s*bomb",
    # While-true damage loops
    r"while\s+true\s*;.*(?:rm|dd|mkfs|shred)",
    r"while\s*:\s*;.*(?:rm|dd|mkfs)",
    # Reverse shells
    r"nc\s+.*-e\s+/bin/(?:sh|bash)",
    r"python.*socket.*connect.*\(.*\d{1,3}\.\d{1,3}",
    # Stealth / history wipe
    r"history\s+-c\s*;",
    r"unset\s+HISTFILE",
]

# ── Medium-risk: warned ──────────────────────────────────────
MEDIUM_RISK_PATTERNS: list[str] = [
    r"chmod\s+777",
    r"chmod\s+-R\s+777",
    r"sudo\b(?!\s+su\b)",     # sudo allowed, sudo su blocked (high-risk)
    r"pip\s+install\b",
    r"pip3\s+install\b",
    r"apt\s+install\b",
    r"apt-get\s+install\b",
    r"npm\s+install\s+-g",
    r"yum\s+install\b",
    r"dnf\s+install\b",
    r"brew\s+install\b",
    r"PATH\s*=",
    r"chown\b(?!\s+root)",   # chown allowed, chown root blocked
    r"kill\s+-9",
]


class CommandAudit:
    """Classify shell commands by risk level."""

    Risk = str  # "block" | "warn" | "pass"

    @classmethod
    def audit(cls, command: str) -> tuple[str, str]:
        """Audit a command. Returns (risk, reason)."""
        if not command or not isinstance(command, str):
            return "pass", ""

        # Input validation
        if len(command) > MAX_COMMAND_LENGTH:
            return "block", "command too long"
        if "\x00" in command:
            return "block", "null byte in command"

        # Pass 1: whole-command scan
        for pattern in HIGH_RISK_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return "block", f"high-risk pattern matched: {pattern}"

        # Pass 2: per-sub-command scan
        sub_commands = cls._split_commands(command)
        for sub in sub_commands:
            sub = sub.strip()
            if not sub:
                continue
            for pattern in HIGH_RISK_PATTERNS:
                if re.search(pattern, sub, re.IGNORECASE):
                    return "block", f"high-risk in sub-command: {pattern}"

            for pattern in MEDIUM_RISK_PATTERNS:
                if re.search(pattern, sub, re.IGNORECASE):
                    return "warn", f"medium-risk pattern: {pattern}"

        return "pass", ""

    @staticmethod
    def _split_commands(command: str) -> list[str]:
        """Quote-aware split on ;, &&, ||."""
        parts: list[str] = []
        current: list[str] = []
        in_single = False
        in_double = False

        i = 0
        while i < len(command):
            c = command[i]

            if c == "'" and not in_double:
                in_single = not in_single
                current.append(c)
            elif c == '"' and not in_single:
                in_double = not in_double
                current.append(c)
            elif not in_single and not in_double:
                if command[i:i+2] == "&&":
                    parts.append("".join(current).strip())
                    current = []
                    i += 2
                    continue
                elif command[i:i+2] == "||":
                    parts.append("".join(current).strip())
                    current = []
                    i += 2
                    continue
                elif c == ";":
                    parts.append("".join(current).strip())
                    current = []
                    i += 1
                    continue
                else:
                    current.append(c)
            else:
                current.append(c)
            i += 1

        if current:
            parts.append("".join(current).strip())

        return [p for p in parts if p]

    # ── User-facing messages ────────────────────────────────

    BLOCK_MESSAGE = "Command blocked — high-risk operation detected."

    @classmethod
    def warn_message(cls, command: str, reason: str) -> str:
        return (
            f"[WARNING] Potentially risky operation: {reason}. "
            "Please use a safer alternative if possible."
        )


# ============================================================================
# Sandbox Audit Middleware
# ============================================================================


class SandboxAuditMiddleware(Middleware):
    """Audit terminal/bash tool calls for dangerous commands.

    Wraps `terminal` and `bash` tool calls.
    High-risk: blocked with error message.
    Medium-risk: allowed with warning appended to result.

    Config:
        audit_tools: list[str] — tool names to audit (default: ["terminal", "bash"])
        block_high: bool — block high-risk commands (default: True)
        warn_medium: bool — append warnings for medium-risk (default: True)
        max_command_length: int — max command string length (default: 10000)
    """

    def __init__(
        self,
        audit_tools: list[str] | None = None,
        block_high: bool = True,
        warn_medium: bool = True,
        max_command_length: int = MAX_COMMAND_LENGTH,
    ):
        self._audit_tools = audit_tools or ["terminal", "bash"]
        self._block_high = block_high
        self._warn_medium = warn_medium
        self._max_command_length = max_command_length
        self.yolo_bypass: bool = False  # /yolo — skip all checks

    def wrap_tool_call(self, tool_name: str, args: dict, handler, **kwargs) -> Any:
        if tool_name not in self._audit_tools:
            return handler(tool_name, args, **kwargs)

        # YOLO mode — pass everything through
        if self.yolo_bypass:
            return handler(tool_name, args, **kwargs)

        command = args.get("command", "")
        if not command:
            return handler(tool_name, args, **kwargs)

        risk, reason = CommandAudit.audit(command)

        if risk == "block" and self._block_high:
            logger.warning(
                "SandboxAudit: blocked %s — %s — cmd: %s",
                tool_name, reason, command[:200],
            )
            return f"BLOCKED: {CommandAudit.BLOCK_MESSAGE} ({reason})"

        result = handler(tool_name, args, **kwargs)

        if risk == "warn" and self._warn_medium:
            logger.info(
                "SandboxAudit: warned %s — %s — cmd: %s",
                tool_name, reason, command[:200],
            )
            warning = CommandAudit.warn_message(command, reason)
            if isinstance(result, str):
                result = f"{warning}\n{result}"
            elif isinstance(result, dict):
                result = {**result, "sandbox_audit_warning": warning}

        return result

    def __repr__(self) -> str:
        return (
            f"SandboxAuditMiddleware(tools={self._audit_tools}, "
            f"block={self._block_high}, warn={self._warn_medium})"
        )
