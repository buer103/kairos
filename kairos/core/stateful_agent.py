"""Stateful Agent — multi-turn conversation with session persistence + streaming.

Supports pluggable session backends:
    - FileSessionBackend (default, local JSON)
    - RedisSessionBackend (production, multi-process)
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Generator

from kairos.core.loop import Agent
from kairos.core.session_backends import FileSessionBackend, SessionBackend
from kairos.core.state import Case, ThreadState


class StatefulAgent(Agent):
    """Multi-turn agent that maintains conversation state across calls.

    Inherits budget control, interrupt, and checkpoint from Agent.
    Adds session persistence (file or Redis) and streaming.

    Args:
        session_backend: Pluggable persistence backend. Defaults to FileSessionBackend.
        session_id: Explicit session ID. Auto-generated if omitted.
    """

    def __init__(
        self,
        *args,
        session_id: str | None = None,
        session_backend: SessionBackend | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._session_id = session_id or str(uuid.uuid4())[:8]
        self._state: ThreadState | None = None
        self._runtime: dict[str, Any] = {}
        self._turn_count = 0
        self._auto_save = True
        self._backend = session_backend or FileSessionBackend()

    # ── Properties ──────────────────────────────────────────────

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

    @property
    def session_backend(self) -> SessionBackend:
        return self._backend

    # ── Chat ────────────────────────────────────────────────────

    def chat(self, user_message: str) -> dict[str, Any]:
        if self.interrupted:
            return {"content": "[Interrupted]", "confidence": None, "evidence": []}

        self._turn_count += 1

        if self._state is None:
            self._state = self._init_conversation(user_message)
        else:
            self._state.messages.append({"role": "user", "content": user_message})

        self._runtime["turn"] = self._turn_count
        result = self._execute_loop(self._state, self._runtime)
        self._auto_save_session()
        return result

    def chat_stream(self, user_message: str) -> Generator[dict, None, None]:
        if self.interrupted:
            yield {"type": "error", "message": "Conversation interrupted."}
            return

        self._turn_count += 1

        if self._state is None:
            self._state = self._init_conversation(user_message)
        else:
            self._state.messages.append({"role": "user", "content": user_message})

        self._runtime["turn"] = self._turn_count
        yield from self._execute_loop_stream(self._state, self._runtime)
        self._auto_save_session()

    def reset(self) -> None:
        self._state = None
        self._runtime = {}
        self._turn_count = 0
        self._session_id = str(uuid.uuid4())[:8]

    # ── Session Persistence ──────────────────────────────────────

    def save_session(self, name: str | None = None) -> None:
        """Save current conversation to the session backend."""
        if self._state is None:
            raise ValueError("No active conversation to save.")
        save_name = name or self._session_id
        data = self._pack_session(save_name)
        self._backend.save(self._session_id, save_name, data)

    def load_session(self, name: str) -> bool:
        """Load a conversation from the session backend."""
        data = self._backend.load(name)
        if data is None:
            return False
        self._unpack_session(data)
        return True

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all saved sessions from the backend."""
        return self._backend.list_sessions()

    def delete_session(self, name: str) -> bool:
        """Delete a saved session from the backend."""
        return self._backend.delete(name)

    # ── Internal ─────────────────────────────────────────────────

    def _pack_session(self, name: str) -> dict[str, Any]:
        """Serialize current state into a JSON-safe dict."""
        data: dict[str, Any] = {
            "session_id": self._session_id,
            "name": name,
            "saved_at": time.time(),
            "turn_count": self._turn_count,
            "messages": self._state.messages,
            "metadata": dict(self._state.metadata or {}),
        }
        if self._state.case:
            data["case"] = {
                "id": self._state.case.id,
                "confidence": self._state.case.confidence,
                "steps": [
                    {"id": s.id, "tool": s.tool, "args": s.args,
                     "result": s.result, "duration_ms": s.duration_ms}
                    for s in self._state.case.steps
                ],
            }
        return data

    def _unpack_session(self, data: dict[str, Any]) -> None:
        """Restore state from a loaded session dict."""
        self._session_id = data["session_id"]
        self._turn_count = data.get("turn_count", 0)
        self._state = ThreadState()
        self._state.messages = data.get("messages", [])
        self._state.metadata = data.get("metadata", {})
        cd = data.get("case", {})
        if cd.get("id"):
            self._state.case = Case(id=cd["id"])
            self._state.case.confidence = cd.get("confidence")
            for s in cd.get("steps", []):
                step = self._state.case.add_step(s["tool"], s["args"])
                if hasattr(self._state.case, "complete_step"):
                    self._state.case.complete_step(
                        step, s.get("result"), s.get("duration_ms", 0))

    def _init_conversation(self, user_message: str) -> ThreadState:
        case = Case(id=f"{self._session_id}_{self._turn_count}")
        state = ThreadState(case=case)
        self._runtime = {
            "user_message": user_message, "session_id": self._session_id,
            "thread_id": self._session_id, "turn": self._turn_count,
        }
        state.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        self.budget.iterations = 0
        self.budget.tokens_used = 0
        self.budget.grace_call_used = False
        self._interrupted = False
        self.pipeline.before_agent(state, self._runtime)
        return state

    def _auto_save_session(self) -> None:
        if self._auto_save and self._state:
            try:
                self.save_session()
            except Exception:
                pass

    def _execute_loop_stream(self, state, runtime) -> Generator[dict, None, None]:
        """Streaming agent loop — yields provider events in real-time."""
        import json as _json
        from kairos.tools.registry import get_tool_schemas, execute_tool

        messages = state.messages

        while not self.budget.exhausted or self.budget.can_grace_call:
            if self._interrupted:
                yield {"type": "done", "content": "[Interrupted]", "tool_calls": None, "usage": {}}
                return

            self.pipeline.before_model(state, runtime)
            tool_schemas = get_tool_schemas() or None

            stream = self.model.chat_stream(messages, tools=tool_schemas)
            assist_msg = {"role": "assistant", "content": ""}
            tool_calls: list[dict] = []

            for event in stream:
                yield event
                if event["type"] == "done":
                    assist_msg["content"] = event.get("content", "")
                    if event.get("tool_calls"):
                        tool_calls = event["tool_calls"]

            self.pipeline.after_model(state, runtime)

            if tool_calls:
                import json as _json_smart
                from kairos.tools.registry import execute_tools_smart

                # Smart dispatch: parallel when safe
                smart_results = execute_tools_smart(tool_calls)

                for i, tc in enumerate(tool_calls):
                    name = tc["name"]
                    try:
                        args_raw = tc.get("arguments", tc.get("args", "{}"))
                        if isinstance(args_raw, str):
                            args = _json_smart.loads(args_raw)
                        else:
                            args = args_raw
                    except Exception:
                        args = {}
                    result = self.pipeline.wrap_tool_call(
                        name, args, lambda n, a, **kw: smart_results[i], state=state,
                    )
                    messages.append({
                        "role": "assistant", "tool_calls": [{
                            "id": tc["id"], "type": "function",
                            "function": {"name": name, "arguments": tc.get("arguments", "{}")},
                        }],
                    })
                    messages.append({
                        "role": "tool", "tool_call_id": tc["id"],
                        "content": _json_smart.dumps(result, ensure_ascii=False),
                    })
                self.budget.step()
                continue

            messages.append(assist_msg)
            self.pipeline.after_agent(state, runtime)
            return

        yield {"type": "done", "content": "Budget exhausted.", "tool_calls": None, "usage": {}}
