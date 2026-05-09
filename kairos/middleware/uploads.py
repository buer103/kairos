"""Uploads middleware — injects uploaded file information into the message context.

Scans the thread's uploads directory and formats file paths into the user message
so the model knows which files are available.

DeerFlow layer 2 — runs after ThreadDataMiddleware, depends on uploads_dir.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from kairos.core.middleware import Middleware


class UploadsMiddleware(Middleware):
    """Injects available file paths into the system/user message context.

    Hook: before_agent — scans uploads directory and formats file info.

    Format:
      <uploaded_files>
      /path/to/thread/uploads/report.pdf (1.2 MB)
      /path/to/thread/uploads/data.csv (340 KB)
      </uploaded_files>
    """

    def before_agent(self, state: Any, runtime: dict[str, Any]) -> dict[str, Any] | None:
        thread_data = state.metadata.get("thread_data", {})
        uploads_dir = thread_data.get("uploads_dir", "")
        if not uploads_dir or not os.path.isdir(uploads_dir):
            return None

        files = self._scan_uploads(uploads_dir)
        if not files:
            return None

        file_block = "<uploaded_files>\n"
        for f in files:
            size_str = self._format_size(f["size"])
            file_block += f"{f['path']} ({size_str})\n"
        file_block += "</uploaded_files>"

        # Inject into the last user message (or system message)
        messages = getattr(state, "messages", [])
        if messages:
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user":
                    messages[i]["content"] = f"{file_block}\n\n{messages[i]['content']}"
                    break
            else:
                # No user message found, inject into system
                if messages[0].get("role") == "system":
                    messages[0]["content"] += f"\n\n{file_block}"

        return None

    @staticmethod
    def _scan_uploads(uploads_dir: str) -> list[dict]:
        """Scan the uploads directory for files."""
        files = []
        try:
            for entry in os.scandir(uploads_dir):
                if entry.is_file() and not entry.name.startswith("."):
                    files.append({
                        "path": entry.path,
                        "name": entry.name,
                        "size": entry.stat().st_size,
                    })
        except OSError:
            pass
        return sorted(files, key=lambda f: f["name"])

    @staticmethod
    def _format_size(size: int) -> str:
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        else:
            return f"{size / (1024 * 1024):.1f} MB"

    def __repr__(self) -> str:
        return "UploadsMiddleware()"
