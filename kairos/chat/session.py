"""Session management — persist and resume agent conversation sessions."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class SessionStore:
    """Persistent session storage.

    Sessions are stored as JSON files under ~/.kairos/sessions/.
    Each session records the message history and metadata for resumption.
    """

    def __init__(self, base_path: str | Path | None = None):
        self._base = Path(base_path or Path.home() / ".kairos" / "sessions")
        self._base.mkdir(parents=True, exist_ok=True)

    def save(self, session_id: str, messages: list[dict], metadata: dict | None = None) -> Path:
        """Save a session to disk."""
        data = {
            "id": session_id,
            "created_at": datetime.now().isoformat(),
            "messages": [
                {
                    "role": m.get("role", ""),
                    "content": m.get("content", ""),
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"],
                            },
                        }
                        for tc in m.get("tool_calls", [])
                    ] if m.get("tool_calls") else None,
                    "tool_call_id": m.get("tool_call_id"),
                }
                for m in messages
            ],
            "metadata": metadata or {},
        }
        path = self._base / f"{session_id}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        return path

    def load(self, session_id: str) -> dict[str, Any] | None:
        """Load a session from disk."""
        path = self._base / f"{session_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        # Clean up None tool_calls fields
        for m in data.get("messages", []):
            if m.get("tool_calls") is None:
                del m["tool_calls"]
            if m.get("tool_call_id") is None:
                del m["tool_call_id"]
        return data

    def delete(self, session_id: str) -> bool:
        """Delete a session."""
        path = self._base / f"{session_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recent sessions."""
        files = sorted(self._base.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        results = []
        for f in files[:limit]:
            data = json.loads(f.read_text())
            results.append({
                "id": data.get("id", f.stem),
                "created_at": data.get("created_at"),
                "message_count": len(data.get("messages", [])),
                "metadata": data.get("metadata", {}),
            })
        return results

    def resume(self, session_id: str, user_message: str) -> list[dict] | None:
        """Resume a session: load messages and append the new user message."""
        session = self.load(session_id)
        if not session:
            return None
        messages = session["messages"]
        messages.append({"role": "user", "content": user_message})
        return messages
