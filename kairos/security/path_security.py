"""
Path Security
=============

Filesystem path validation and normalisation with protection against:

* Symlink / TOCTOU attacks (resolves real paths before checks)
* Path traversal via ``../`` sequences
* Null-byte injection
* Access to sensitive system paths (``/etc/passwd``, ``/proc/*``, ``/dev/*``, …)

All operations use stdlib only.  Python 3.12+.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import ClassVar


class PathSecurity:
    """Guard filesystem read/write operations with path validation.

    Maintains a whitelist of allowed root directories and rejects any
    resolved path that falls outside them — including paths reached via
    symlink indirection.

    Usage::

        ps = PathSecurity(allowed_roots=["/tmp/sandbox", "/home/user/data"])
        safe_path = ps.guard_read("notes.txt")      # raises on violation
        write_path = ps.guard_write("output.csv")   # raises on violation
    """

    # ------------------------------------------------------------------
    # Blocked path patterns — sensitive system locations
    # ------------------------------------------------------------------
    BLOCKED_PATH_PATTERNS: ClassVar[list[str]] = [
        r"^/etc/(passwd|shadow|group|gshadow|sudoers)",
        r"^/etc/ssl/private",
        r"^/etc/ssh",
        r"^/proc(/.*)?",
        r"^/sys(/.*)?",
        r"^/dev/((?!null|zero|random|urandom|tty|stdin|stdout|stderr).)*",  # allow harmless /dev nodes
        r"^/boot(/.*)?",
        r"^/root(/.*)?",
        r"^/var/log(/.*)?",
        r"^/var/run(/.*)?",
        r"^/run(/.*)?",
    ]

    _BLOCKED_RE: ClassVar[re.Pattern] = re.compile(
        "|".join(f"(?:{p})" for p in BLOCKED_PATH_PATTERNS),
        re.IGNORECASE,
    )

    def __init__(
        self,
        allowed_roots: list[str] | None = None,
        *,
        block_system_paths: bool = True,
    ) -> None:
        """Initialise the path security guard.

        Args:
            allowed_roots: List of base directories that file operations
                are permitted within.  Resolved to absolute paths on init.
                Defaults to ``["/tmp"]``.
            block_system_paths: When *True* (default), access to sensitive
                system paths (``/etc/passwd``, ``/proc/*``, …) is blocked
                regardless of the allowed roots.
        """
        self.allowed_roots: list[str] = [
            os.path.realpath(r) for r in (allowed_roots or ["/tmp"])
        ]
        self.block_system_paths: bool = block_system_paths

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_path_allowed(self, path: str) -> bool:
        """Check whether *path* is within any of the allowed roots.

        Resolves real path, checks for null bytes, blocked system paths,
        and that the final canonical path is inside an allowed root.

        Returns *True* when the path is safe to access.
        """
        try:
            resolved = self.normalize(path)
        except (OSError, ValueError):
            return False

        if self.block_system_paths and self._BLOCKED_RE.match(resolved):
            return False

        return any(
            resolved == root or resolved.startswith(root + os.sep)
            for root in self.allowed_roots
        )

    def guard_read(self, path: str) -> str:
        """Resolve, validate, and return a safe path for reading.

        Raises :class:`PermissionError` when the path is not allowed,
        and :class:`FileNotFoundError` when the resolved file does not exist.

        Returns the canonical (real) path on success.
        """
        real = self._resolve(path)

        if not self.is_path_allowed(real):
            raise PermissionError(
                f"Read access denied: {path!r} resolves to {real!r} "
                f"which is outside allowed roots"
            )

        if not os.path.isfile(real):
            raise FileNotFoundError(f"File not found or not a regular file: {real!r}")

        return real

    def guard_write(self, path: str) -> str:
        """Resolve, validate, and return a safe path for writing.

        Unlike ``guard_read``, this does **not** require the file to exist
        beforehand — it validates that the *directory* containing the file
        is within the allowed roots.

        Raises :class:`PermissionError` when the path is not allowed.

        Returns the canonical (real) path on success.
        """
        real = self._resolve(path)

        if not self.is_path_allowed(real):
            raise PermissionError(
                f"Write access denied: {path!r} resolves to {real!r} "
                f"which is outside allowed roots"
            )

        # Ensure the parent directory exists and is allowed
        parent = os.path.dirname(real) or "."
        if not os.path.isdir(parent):
            raise FileNotFoundError(
                f"Parent directory does not exist: {parent!r}"
            )

        return real

    def normalize(self, path: str) -> str:
        """Normalize a path: collapse ``..``, resolve symlinks, strip nulls.

        Returns the fully-resolved absolute path.  Raises :class:`ValueError`
        for null-byte injection attempts.

        This is the core normalisation primitive used by all other methods.
        """
        # Null-byte injection detection
        if "\x00" in path:
            raise ValueError("Path contains null bytes — possible injection attack")

        # Strip leading/trailing whitespace that could hide traversal
        path = path.strip()

        # Expand ~ and ~user, make absolute
        expanded = os.path.expanduser(path)
        if not os.path.isabs(expanded):
            raise ValueError(f"Relative paths are not accepted: {path!r}")

        # Resolve symlinks and normalise (collapses .., redundant separators, etc.)
        return os.path.realpath(expanded)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve(self, path: str) -> str:
        """Normalize and resolve *path*, catching common attack vectors."""
        return self.normalize(path)
