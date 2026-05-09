"""
Content Redactor
================

Scrubs sensitive information from text and dictionaries:

* OpenAI / LLM API keys (``sk-*``, ``sk-proj-*``)
* GitHub tokens (``ghp_*``, ``github_pat_*``)
* Generic ``Bearer`` / authorization tokens
* API keys in ``key=value`` or JSON-like formats
* Password fields in dictionaries

Zero external dependencies.  Python 3.12+.
"""

from __future__ import annotations

import re
from typing import ClassVar


_REDACTED_MARKER: str = "[REDACTED]"


class ContentRedactor:
    """Detect and redact sensitive secrets from strings and dictionaries.

    Usage::

        redactor = ContentRedactor()
        clean = redactor.redact("My key is sk-abc123xyz")
        # "My key is [REDACTED]"

        data = {"auth": {"api_key": "secret"}, "name": "Alice"}
        clean_data = redactor.redact_dict(data)
        # {"auth": {"api_key": "[REDACTED]"}, "name": "Alice"}
    """

    # ------------------------------------------------------------------
    # Regex patterns for known secret formats
    # ------------------------------------------------------------------

    # OpenAI keys: sk-<project>-<random> or sk-<random>
    _OPENAI_KEY_RE: ClassVar[re.Pattern] = re.compile(
        r"\b(sk-(?:proj-)?[A-Za-z0-9_\-]{20,})\b"
    )

    # GitHub classic tokens: ghp_<40 chars>
    _GITHUB_CLASSIC_RE: ClassVar[re.Pattern] = re.compile(
        r"\b(ghp_[A-Za-z0-9]{36,40})\b"
    )

    # GitHub fine-grained tokens: github_pat_<...>
    _GITHUB_PAT_RE: ClassVar[re.Pattern] = re.compile(
        r"\b(github_pat_[A-Za-z0-9_\-]{20,})\b"
    )

    # Generic Bearer token patterns
    _BEARER_RE: ClassVar[re.Pattern] = re.compile(
        r"(?:Bearer|bearer)\s+([A-Za-z0-9_\-\.=]{20,})",
        re.IGNORECASE,
    )

    # "key=value" style secrets (looking for typical secret key names)
    _KEY_VALUE_SECRET_RE: ClassVar[re.Pattern] = re.compile(
        r"(?:api[_-]?key|apikey|secret|token|password|passwd|auth)\s*[:=]\s*"
        r"[\"']?([A-Za-z0-9_\-\.=+/]{16,})[\"']?",
        re.IGNORECASE,
    )

    # AWS-style keys: AKIA..., ASIA...
    _AWS_KEY_RE: ClassVar[re.Pattern] = re.compile(
        r"\b(AKIA|ASIA)[A-Z0-9]{16}\b"
    )

    # Generic high-entropy base64-like strings that look like secrets
    _HIGH_ENTROPY_BASE64_RE: ClassVar[re.Pattern] = re.compile(
        r"\b([A-Za-z0-9+/]{40,}={0,2})\b"
    )

    # List of (name, pattern, replacement_template) for ordered application
    _PATTERNS: ClassVar[list[tuple[str, re.Pattern, str]]] = [
        ("openai-key", _OPENAI_KEY_RE, _REDACTED_MARKER),
        ("github-classic", _GITHUB_CLASSIC_RE, _REDACTED_MARKER),
        ("github-pat", _GITHUB_PAT_RE, _REDACTED_MARKER),
        ("aws-key", _AWS_KEY_RE, _REDACTED_MARKER),
        ("bearer-token", _BEARER_RE, f"Bearer {_REDACTED_MARKER}"),
        ("key-value-secret", _KEY_VALUE_SECRET_RE, _REDACTED_MARKER),
    ]

    # Dictionary keys that suggest sensitive values
    SENSITIVE_DICT_KEYS: ClassVar[frozenset[str]] = frozenset({
        "api_key", "apikey", "api_secret",
        "secret", "token", "access_token", "refresh_token",
        "password", "passwd", "pwd",
        "authorization", "auth",
        "private_key", "privatekey",
        "credential", "credentials",
        "ssh_key", "sshkey",
        "db_password", "db_pass",
        "connection_string",
    })

    def __init__(self, *, custom_patterns: list[tuple[str, str]] | None = None) -> None:
        """Initialise the redactor.

        Args:
            custom_patterns: Optional list of ``(name, regex_string)`` pairs
                for additional secret patterns to detect.
        """
        self.patterns: list[tuple[str, re.Pattern, str]] = list(self._PATTERNS)

        if custom_patterns:
            for name, regex_str in custom_patterns:
                self.patterns.append(
                    (name, re.compile(regex_str), _REDACTED_MARKER)
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def redact(self, text: str) -> str:
        """Redact sensitive secrets from a text string.

        Args:
            text: The input string that may contain secrets.

        Returns:
            The text with all recognised secrets replaced by ``[REDACTED]``.
        """
        result = text
        for _name, pattern, replacement in self.patterns:
            result = pattern.sub(replacement, result)
        return result

    def redact_dict(self, data: dict) -> dict:
        """Recursively redact sensitive values in a dictionary.

        Keys are matched against :attr:`SENSITIVE_DICT_KEYS` (case-insensitive).
        Values that are strings are passed through :meth:`redact`.
        Nested dicts and lists are traversed recursively.

        Args:
            data: The input dictionary (may contain nested dicts/lists).

        Returns:
            A new dictionary with sensitive values redacted.
        """
        return self._redact_value(data)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _redact_value(self, value: object) -> object:
        """Recursively redact any value (dict, list, str, or scalar)."""
        match value:
            case dict():
                return {
                    k: _REDACTED_MARKER
                    if self._is_sensitive_key(k)
                    else self._redact_value(v)
                    for k, v in value.items()
                }
            case list():
                return [self._redact_value(item) for item in value]
            case str():
                return self.redact(value)
            case _:
                return value

    def _is_sensitive_key(self, key: object) -> bool:
        """Check whether a dictionary key implies a sensitive value."""
        if not isinstance(key, str):
            return False
        # Normalize: lowercase, strip underscores and hyphens
        normalized = key.lower().replace("_", "").replace("-", "")
        return normalized in self.SENSITIVE_DICT_KEYS
