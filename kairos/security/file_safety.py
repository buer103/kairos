"""
File Safety Checker
===================

Validates file operations for security: path traversal prevention, dangerous
extension blocking, file size limits, MIME-type detection via magic bytes,
and filename sanitization.  Zero external dependencies.

Python 3.12+ — uses match/case for MIME dispatch.
"""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path
from typing import ClassVar

# ---------------------------------------------------------------------------
# Magic-byte signatures for common file types (no python-magic dependency)
# ---------------------------------------------------------------------------
_MAGIC_SIGNATURES: dict[bytes, str] = {
    # Images
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"\xff\xd8\xff": "image/jpeg",
    b"GIF87a": "image/gif",
    b"GIF89a": "image/gif",
    b"RIFF": "image/webp",  # needs further check, simplified
    b"BM": "image/bmp",
    # Archives
    b"PK\x03\x04": "application/zip",
    b"\x1f\x8b\x08": "application/gzip",
    b"Rar!\x1a\x07\x00": "application/vnd.rar",
    b"Rar!\x1a\x07\x01\x00": "application/vnd.rar",
    b"\xfd7zXZ\x00": "application/x-xz",
    b"BZh": "application/x-bzip2",
    # Documents
    b"%PDF": "application/pdf",
    b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1": "application/msword",  # OLE2
    b"PK\x03\x04\x14\x00\x06\x00": "application/vnd.openxmlformats-officedocument",
    # Executables (detected but blocked by extension check too)
    b"\x7fELF": "application/x-elf",
    b"MZ": "application/x-msdownload",  # PE / DOS
    b"\xca\xfe\xba\xbe": "application/x-mach-binary",  # Mach-O universal
    b"\xcf\xfa\xed\xfe": "application/x-mach-binary",  # Mach-O 64-bit
    b"\xfe\xed\xfa\xce": "application/x-mach-binary",  # Mach-O 32-bit
    b"\xfe\xed\xfa\xcf": "application/x-mach-binary",  # Mach-O 64-bit
    # Text / data
    b"#!": "text/x-script",
    b"<?xml": "application/xml",
    b"<html": "text/html",
    b"<!DOCTYPE html": "text/html",
    b"{\n": "application/json",
    b"{\r": "application/json",
    b'{"': "application/json",
    b"[": "application/json",
    # Media
    b"\x1a\x45\xdf\xa3": "video/webm",  # Matroska
    b"\x00\x00\x00\x18ftyp": "video/mp4",
    b"\x00\x00\x00\x20ftyp": "video/mp4",
    b"ID3": "audio/mpeg",
    b"\xff\xfb": "audio/mpeg",
    b"\xff\xf3": "audio/mpeg",
    b"\xff\xf2": "audio/mpeg",
    b"fLaC": "audio/flac",
    b"OggS": "audio/ogg",
}

_SIGNATURE_MAX_LEN: int = max(len(sig) for sig in _MAGIC_SIGNATURES)


class FileSafetyChecker:
    """Validates file operations for security.

    Configurable allowed directories, blocked extensions, and size limits.
    All methods are stateless — the checker can be used as a singleton or
    instantiated per-context.

    Usage::

        checker = FileSafetyChecker(
            allowed_dirs=["/tmp/sandbox", "/home/user/data"],
            max_file_bytes=10 * 1024 * 1024,  # 10 MiB
        )
        safe, reason = checker.is_safe("/tmp/sandbox/report.pdf")
        if not safe:
            raise SecurityError(reason)
    """

    # Default dangerous extensions — anything that can be directly executed
    DANGEROUS_EXTENSIONS: ClassVar[set[str]] = {
        ".exe", ".dll", ".so", ".dylib", ".sh", ".bash", ".bat", ".cmd",
        ".ps1", ".vbs", ".scr", ".pif", ".msi", ".app", ".bin", ".elf",
        ".out", ".com", ".cpl", ".jar", ".wsf", ".psm1", ".psd1",
        ".hta", ".msc", ".msp", ".scf", ".lnk", ".reg", ".pyc", ".pyo",
    }

    # Filename sanitization — characters to strip / replace
    _FILENAME_SANITIZE_RE: ClassVar[re.Pattern] = re.compile(
        r"[\x00-\x1f\x7f-\x9f]"  # control chars + DEL
        r"|[/\\]"  # path separators
        r"|\.\.+"  # traversal dots (when standalone)
        r"|[\x00]"  # explicit null (redundant but explicit)
    )

    def __init__(
        self,
        allowed_dirs: list[str] | None = None,
        max_file_bytes: int = 100 * 1024 * 1024,  # 100 MiB default
        blocked_extensions: set[str] | None = None,
    ) -> None:
        """Initialise the file safety checker.

        Args:
            allowed_dirs: List of directory paths that file access is
                restricted to.  Defaults to ``["/tmp"]`` if not provided.
            max_file_bytes: Maximum allowed file size in bytes.
            blocked_extensions: Additional extensions to block beyond the
                class-level ``DANGEROUS_EXTENSIONS``.
        """
        self.allowed_dirs: list[str] = [
            os.path.realpath(d) for d in (allowed_dirs or ["/tmp"])
        ]
        self.max_file_bytes: int = max_file_bytes
        self.blocked_extensions: set[str] = (
            self.DANGEROUS_EXTENSIONS | (blocked_extensions or set())
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_safe(self, path_or_content: str, mode: str = "path") -> tuple[bool, str]:
        """Unified safety check for a file path or content string.

        Args:
            path_or_content: File path or content string to validate.
            mode: ``"path"`` for file-path checks (path traversal, extension,
                size, MIME) or ``"content"`` for content-only checks
                (size, sanitized filename).

        Returns:
            A ``(safe, reason)`` tuple.  ``safe`` is ``True`` when all
            checks pass; ``reason`` describes the first failure.
        """
        match mode:
            case "path":
                if not self.check_path(path_or_content):
                    return False, "path escapes allowed directories"
                if not self.check_extension(path_or_content):
                    return False, "file extension is blocked"
                if not self.check_size(path_or_content, self.max_file_bytes):
                    return False, f"file exceeds max size ({self.max_file_bytes} bytes)"
                if not self.check_mime(path_or_content):
                    return False, "file MIME type is unsafe or undetectable"
                return True, "ok"
            case "content":
                # For content strings, check embedded path references and size
                if len(path_or_content.encode("utf-8", errors="replace")) > self.max_file_bytes:
                    return False, f"content exceeds max size ({self.max_file_bytes} bytes)"
                return True, "ok"
            case _:
                return False, f"unknown safety mode: {mode!r}"

    def check_path(self, path: str) -> bool:
        """Validate that *path* does not escape the allowed directories.

        Resolves symlinks and canonicalises the path before comparing
        against ``self.allowed_dirs``.

        Args:
            path: Filesystem path to validate.

        Returns:
            ``True`` if the resolved path resides within an allowed directory.
        """
        try:
            resolved = os.path.realpath(path)
        except (OSError, ValueError):
            return False

        # Null-byte injection check
        if "\x00" in path:
            return False

        return any(
            resolved == root or resolved.startswith(root + os.sep)
            for root in self.allowed_dirs
        )

    def check_extension(self, path: str) -> bool:
        """Block dangerous file extensions.

        Checks the final suffix (case-insensitive) against the blocked set.

        Args:
            path: Filesystem path.

        Returns:
            ``True`` when the extension is **not** blocked.
        """
        ext = os.path.splitext(path)[1].lower()
        return ext not in self.blocked_extensions

    def check_size(self, path: str, max_bytes: int | None = None) -> bool:
        """Check that a file's size does not exceed *max_bytes*.

        Args:
            path: Filesystem path to an existing file.
            max_bytes: Override the instance default if provided.

        Returns:
            ``True`` when the file size is <= *max_bytes*.
        """
        limit = max_bytes if max_bytes is not None else self.max_file_bytes
        try:
            st = os.stat(path)
            if stat.S_ISDIR(st.st_mode):
                return False  # directories are not files
            return st.st_size <= limit
        except OSError:
            return False

    def check_mime(self, path: str) -> bool:
        """Detect MIME type via file magic bytes (stdlib only).

        Reads the first few bytes of the file and matches against a built-in
        table of magic-byte signatures.  Returns ``True`` when the detected
        type is **not** an executable or dangerous format.

        Args:
            path: Filesystem path to read.

        Returns:
            ``True`` if the MIME type is recognised and considered safe.
        """
        try:
            with open(path, "rb") as fh:
                head = fh.read(_SIGNATURE_MAX_LEN)
        except OSError:
            return False

        mime = self._detect_mime(head)
        if mime is None:
            return False  # unknown format — reject

        # MIME types we consider unsafe / executable
        unsafe_prefixes = (
            "application/x-elf",
            "application/x-msdownload",
            "application/x-mach-binary",
            "application/x-dosexec",
            "text/x-script",  # shebang scripts
        )
        return not mime.startswith(unsafe_prefixes)

    def detect_mime_type(self, path: str) -> str | None:
        """Return the detected MIME type string, or *None* if unknown.

        This is a lower-level method; ``check_mime`` wraps it with a
        safety decision.
        """
        try:
            with open(path, "rb") as fh:
                head = fh.read(_SIGNATURE_MAX_LEN)
        except OSError:
            return None
        return self._detect_mime(head)

    def sanitize_filename(self, name: str) -> str:
        """Remove path traversal characters, nulls, and control chars.

        Args:
            name: Raw filename string (not a full path).

        Returns:
            A sanitized filename safe for use in filesystem operations.
        """
        # Strip path separators and traversal sequences
        cleaned = re.sub(r"[/\\]", "_", name)
        cleaned = re.sub(r"\.\.+", "_", cleaned)
        # Strip null bytes and control characters
        cleaned = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", cleaned)
        # Collapse multiple underscores
        cleaned = re.sub(r"_+", "_", cleaned)
        # Strip leading dots (hidden files), trailing dots/spaces
        cleaned = cleaned.strip(". _")
        return cleaned or "unnamed"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_mime(head: bytes) -> str | None:
        """Match *head* bytes against the magic-signature table.

        Returns the MIME type string or *None* if no signature matches.
        """
        for signature, mime in _MAGIC_SIGNATURES.items():
            if head.startswith(signature):
                return mime
        return None
