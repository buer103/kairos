"""Uploads middleware — file injection with type detection and size enforcement.

DeerFlow layer 2 — runs after ThreadDataMiddleware.

Enhancements over v0.8.0:
  - MIME type detection from file extension
  - Preview snippets for text files
  - Size limits (skip files > max_size)
  - Max file count cap
  - Structured XML injection with metadata
"""

from __future__ import annotations

import logging
import mimetypes
import os
from pathlib import Path
from typing import Any

from kairos.core.middleware import Middleware

logger = logging.getLogger("kairos.middleware.uploads")

# Initialize MIME types
mimetypes.init()


class UploadsMiddleware(Middleware):
    """Injects uploaded file paths with metadata into the message context.

    Hook: before_agent — scans uploads directory, formats file info.
    """

    def __init__(
        self,
        max_files: int = 20,
        max_size: int = 50 * 1024 * 1024,  # 50 MB
        include_preview: bool = True,
        preview_bytes: int = 500,
    ):
        self._max_files = max_files
        self._max_size = max_size
        self._include_preview = include_preview
        self._preview_bytes = preview_bytes

    def before_agent(self, state: Any, runtime: dict[str, Any]) -> dict[str, Any] | None:
        td = (state.metadata or {}).get("thread_data", {})
        uploads_dir = td.get("uploads") or td.get("uploads_dir", "")
        if not uploads_dir or not os.path.isdir(uploads_dir):
            return None

        files = self._scan(uploads_dir)
        if not files:
            return None

        block = self._format_block(files)

        messages = getattr(state, "messages", [])
        if messages:
            self._inject(messages, block)

        return None

    def _scan(self, uploads_dir: str) -> list[dict]:
        files = []
        try:
            for entry in os.scandir(uploads_dir):
                if not entry.is_file() or entry.name.startswith("."):
                    continue
                info = entry.stat()
                if info.st_size > self._max_size:
                    logger.debug("Skipping oversized file: %s (%d bytes)", entry.name, info.st_size)
                    continue
                mime, _ = mimetypes.guess_type(entry.name)
                preview = ""
                if self._include_preview and mime and mime.startswith("text/"):
                    try:
                        with open(entry.path, "r", encoding="utf-8", errors="replace") as f:
                            preview = f.read(self._preview_bytes)
                    except Exception:
                        pass
                files.append({
                    "path": entry.path,
                    "name": entry.name,
                    "size": info.st_size,
                    "mime": mime or "application/octet-stream",
                    "preview": preview,
                })
        except OSError:
            pass

        files.sort(key=lambda f: f["name"])
        return files[:self._max_files]

    def _format_block(self, files: list[dict]) -> str:
        lines = ["<uploaded_files>"]
        for f in files:
            size_str = self._fmt_size(f["size"])
            lines.append(f'  <file name="{f["name"]}" size="{size_str}" mime="{f["mime"]}">')
            lines.append(f"    {f['path']}")
            if f["preview"]:
                lines.append(f'    <preview>{f["preview"]}</preview>')
            lines.append("  </file>")
        lines.append("</uploaded_files>")
        return "\n".join(lines)

    @staticmethod
    def _inject(messages: list[dict], block: str) -> None:
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                messages[i]["content"] = f"{block}\n\n{messages[i]['content']}"
                return
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] += f"\n\n{block}"

    @staticmethod
    def _fmt_size(size: int) -> str:
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        return f"{size / (1024 * 1024):.1f} MB"

    def __repr__(self) -> str:
        return f"UploadsMiddleware(max_files={self._max_files}, max_size={self._max_size // (1024*1024)}MB)"
