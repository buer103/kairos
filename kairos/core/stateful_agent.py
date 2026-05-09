"""Stateful Agent — multi-turn conversation with persistent session state."""

from __future__ import annotations

import json
import signal
import time
import uuid
from pathlib import Path
from typing import Any, Generator

from kairos.core.loop import Agent
from kairos.core.state import Case, ThreadState


class StatefulAgent(Agent):
    """Multi-turn agent that maintains conversation state across calls."""

    def __init__(self, *args, session_id: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._session_id = session_id or str(uuid.uuid4())[:8]
        self._state: ThreadState | None = None
        self._runtime: dict[str, Any] = {}
        self._turn_count = 0
        self._interrupted = False
        self._auto_save = True
        self._session_dir = Path.home() / ".kairos" / "sessions"
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._setup_signal_handlers()

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @property
    def history(self) -> list[dict]:
        if self._state is None:
            return []
        return [m for m in self._state.messages if m.get("role") != "system"]

    # ── Public API ──────────────────────────────────────────────

    def chat(self, user_message: str) -> dict[str, Any]:
        """Send a message and get a response, maintaining conversation state."""
        if self._interrupted:
            return {"content": "[Interrupted]", "confidence": None, "evidence": []}

        self._turn_count += 1

        if self._state is None:
            self._state = self._init_conversation(user_message)
        else:
            self._state.messages.append({"role": "user", "content": user_message})

        result = self._execute_loop(self._state, self._runtime)
        self._auto_save_session()
        return result

    def chat_stream(self, user_message: str) -> Generator[dict, None, None]:
        """Chat with streaming — yields response chunks."""
        if self._interrupted:
            yield {"type": "error", "message": "Conversation interrupted."}
            return

        result = self.chat(user_message)
        words = result.get("content", "").split()
        chunk_size = max(1, len(words) // 10) if words else 1

        for i in range(0, len(words), chunk_size):
            yield {"type": "token", "content": " ".join(words[i : i + chunk_size]) + " "}

        yield {
            "type": "done",
            "content": result.get("content", ""),
            "confidence": result.get("confidence"),
            "evidence": result.get("evidence", []),
        }

    def reset(self) -> None:
        self._state = None
        self._runtime = {}
        self._turn_count = 0
        self._interrupted = False
        self._session_id = str(uuid.uuid4())[:8]

    def interrupt(self) -> None:
        self._interrupted = True
        if self._auto_save:
            self._auto_save_session()

    # ── Session Persistence ──────────────────────────────────────

    def save_session(self, name: str | None = None) -> Path:
        if self._state is None:
            raise ValueError("No active conversation to save.")

        save_name = name or self._session_id
        path = self._session_dir / f"{save_name}.json"

        data = {
            "session_id": self._session_id,
            "name": save_name,
            "saved_at": time.time(),
            "turn_count": self._turn_count,
            "messages": self._state.messages,
            "metadata": dict(self._state.metadata),
            "case": {
                "id": self._state.case.id if self._state.case else "",
                "confidence": self._state.case.confidence if self._state.case else None,
                "steps": [
                    {"id": s.id, "tool": s.tool, "args": s.args, "result": s.result, "duration_ms": s.duration_ms}
                    for s in (self._state.case.steps if self._state.case else [])
                ],
            },
        }

        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        return path

    def load_session(self, name: str) -> bool:
        path = self._session_dir / f"{name}.json"
        if not path.exists():
            return False

        data = json.loads(path.read_text())
        self._session_id = data["session_id"]
        self._turn_count = data["turn_count"]

        self._state = ThreadState()
        self._state.messages = data["messages"]
        self._state.metadata = data.get("metadata", {})

        case_data = data.get("case", {})
        if case_data.get("id"):
            self._state.case = Case(id=case_data["id"])
            self._state.case.confidence = case_data.get("confidence")
            for s in case_data.get("steps", []):
                self._state.case.add_step(s["tool"], s["args"])

        self._interrupted = False
        return True

    def list_sessions(self) -> list[dict[str, Any]]:
        sessions = []
        for f in sorted(self._session_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(f.read_text())
                sessions.append({
                    "name": data.get("name", f.stem),
                    "session_id": data.get("session_id"),
                    "saved_at": data.get("saved_at"),
                    "turn_count": data.get("turn_count", 0),
                    "message_count": len(data.get("messages", [])),
                })
            except Exception:
                pass
        return sessions

    def delete_session(self, name: str) -> bool:
        path = self._session_dir / f"{name}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    # ── Internal ─────────────────────────────────────────────────

    def _init_conversation(self, user_message: str) -> ThreadState:
        case = Case(id=f"{self._session_id}_{self._turn_count}")
        state = ThreadState(case=case)
        self._runtime = {
            "user_message": user_message,
            "session_id": self._session_id,
            "thread_id": self._session_id,
            "turn": self._turn_count,
        }
        state.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        self.pipeline.before_agent(state, self._runtime)
        return state

    def _auto_save_session(self) -> None:
        if self._auto_save and self._state:
            try:
                self.save_session()
            except Exception:
                pass

    def _setup_signal_handlers(self) -> None:
        def _handler(signum, frame):
            self._interrupted = True
        try:
            signal.signal(signal.SIGINT, _handler)
        except (ValueError, OSError):
            pass
