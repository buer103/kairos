"""Trajectory recorder — records agent sessions as ShareGPT JSONL for RL training.

Compatible with Atropos and other RL frameworks that consume ShareGPT format.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class TrajectoryRecorder:
    """Records agent conversation trajectories to JSONL files.

    Output format: ShareGPT-compatible JSONL with OpenAI tool call structure.

    Each line contains:
      {
        "id": "traj_xxx",
        "conversations": [
          {"from": "human", "value": "..."},
          {"from": "gpt", "value": "..."},
          ...
        ],
        "tools": [...],
        "metadata": {"confidence": 0.92, "evidence": [...], "duration_ms": 1234}
      }
    """

    def __init__(self, output_dir: str | Path | None = None):
        self._output_dir = Path(output_dir or Path.home() / ".kairos" / "trajectories")
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._buffer: list[dict] = []
        self._trajectory_count = 0

    def record(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Record a complete agent trajectory.

        Args:
            messages: list of role/content dicts from the agent session
            tools: OpenAI-compatible tool schemas used
            metadata: session metadata (confidence, evidence, duration, etc.)
        """
        conversations = []
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "user":
                conversations.append({"from": "human", "value": content})
            elif role == "assistant":
                value = content
                if m.get("tool_calls"):
                    tc_text = json.dumps(m["tool_calls"], ensure_ascii=False)
                    value = f"{content}\n<tool_calls>{tc_text}</tool_calls>" if content else f"<tool_calls>{tc_text}</tool_calls>"
                conversations.append({"from": "gpt", "value": value})
            elif role == "tool":
                conversations.append({"from": "tool", "value": str(content)})
            # system messages are skipped

        entry = {
            "id": f"traj_{self._trajectory_count:06d}",
            "conversations": conversations,
            "tools": tools or [],
            "metadata": metadata or {},
        }
        self._buffer.append(entry)
        self._trajectory_count += 1

    def flush(self, filename: str | None = None) -> Path:
        """Write all buffered trajectories to a JSONL file.

        Args:
            filename: output filename (default: trajectories_<timestamp>.jsonl)

        Returns:
            Path to the written file.
        """
        if not self._buffer:
            return self._output_dir / "empty"

        name = filename or f"trajectories_{int(time.time())}.jsonl"
        path = self._output_dir / name

        with path.open("a", encoding="utf-8") as f:
            for entry in self._buffer:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        self._buffer = []
        return path

    def count(self) -> int:
        """Total trajectories recorded (since last flush)."""
        return len(self._buffer)

    def total_recorded(self) -> int:
        """Total trajectories recorded overall."""
        return self._trajectory_count


class ToolContext:
    """Post-rollout access to the same filesystem/terminal the agent used.

    Reward functions use this to verify agent actions by inspecting the
    filesystem state after a rollout.
    """

    def __init__(self, workdir: str | Path | None = None):
        self.workdir = Path(workdir or Path.cwd())
        self._files_before: dict[str, str] = {}
        self._files_after: dict[str, str] = {}
        self._terminal_outputs: list[str] = []

    def snapshot_before(self) -> None:
        """Capture filesystem state before the agent runs."""
        self._files_before = self._snapshot_files()

    def snapshot_after(self) -> None:
        """Capture filesystem state after the agent runs."""
        self._files_after = self._snapshot_files()

    def record_terminal(self, output: str) -> None:
        """Record a terminal command output."""
        self._terminal_outputs.append(output)

    def file_changed(self, rel_path: str) -> bool:
        """Check if a file was modified by the agent."""
        before = self._files_before.get(rel_path)
        after = self._files_after.get(rel_path)
        return before != after

    def file_created(self, rel_path: str) -> bool:
        """Check if a file was created by the agent."""
        was_there = rel_path in self._files_before
        is_there = rel_path in self._files_after
        return not was_there and is_there

    def grep_output(self, pattern: str) -> bool:
        """Check if any terminal output contains a pattern."""
        return any(pattern in out for out in self._terminal_outputs)

    def _snapshot_files(self) -> dict[str, str]:
        """Get a snapshot of file hashes in the workdir (shallow)."""
        import hashlib
        snapshot = {}
        for f in self.workdir.glob("*"):
            if f.is_file() and not f.name.startswith("."):
                try:
                    h = hashlib.md5(f.read_bytes()).hexdigest()
                    snapshot[f.name] = h
                except Exception:
                    pass
        return snapshot
