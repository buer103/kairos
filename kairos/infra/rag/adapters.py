"""Knowledge base adapters — pluggable connectors for different source formats."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class MarkdownAdapter:
    """Load and chunk Markdown files into documents."""

    @staticmethod
    def load(path: str | Path) -> list[dict[str, Any]]:
        """Load a Markdown file. Returns list of {content, metadata} dicts."""
        p = Path(path)
        if not p.exists():
            return []
        content = p.read_text(encoding="utf-8")
        # Split by headings for rough chunking
        chunks = []
        current_title = ""
        current_lines = []
        for line in content.split("\n"):
            if line.startswith("# "):
                if current_lines:
                    chunks.append({
                        "content": "\n".join(current_lines),
                        "metadata": {"source": str(p), "title": current_title},
                    })
                current_title = line.lstrip("# ").strip()
                current_lines = [line]
            else:
                current_lines.append(line)
        if current_lines:
            chunks.append({
                "content": "\n".join(current_lines),
                "metadata": {"source": str(p), "title": current_title},
            })
        return chunks or [{"content": content, "metadata": {"source": str(p)}}]


class TextAdapter:
    """Load plain text files."""

    @staticmethod
    def load(path: str | Path) -> list[dict[str, Any]]:
        p = Path(path)
        if not p.exists():
            return []
        content = p.read_text(encoding="utf-8")
        return [{"content": content, "metadata": {"source": str(p)}}]
