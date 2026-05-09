"""Thread data middleware — creates workspace directory structure per thread.

DeerFlow layer 1 — must be the FIRST middleware.
Provides workspace/uploads/outputs directories for each conversation thread.
Uses lazy initialization: only creates directories when first needed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from kairos.core.middleware import Middleware


class ThreadDataMiddleware(Middleware):
    """Creates per-thread directory structure.

    Hook: before_agent — computes paths and optionally creates directories.

    Directory layout:
      {base_dir}/threads/{thread_id}/
        ├── workspace/    # Tool code execution sandbox
        ├── uploads/      # User-uploaded files
        └── outputs/      # Tool-produced output files
    """

    def __init__(
        self,
        base_dir: str | Path | None = None,
        lazy_init: bool = True,
    ):
        self._base = Path(base_dir or Path.home() / ".kairos" / "threads")
        self._lazy_init = lazy_init

    def before_agent(self, state: Any, runtime: dict[str, Any]) -> dict[str, Any] | None:
        thread_id = runtime.get("thread_id") or runtime.get("session_id") or "default"

        workspace = self._base / thread_id / "workspace"
        uploads = self._base / thread_id / "uploads"
        outputs = self._base / thread_id / "outputs"

        if not self._lazy_init:
            workspace.mkdir(parents=True, exist_ok=True)
            uploads.mkdir(parents=True, exist_ok=True)
            outputs.mkdir(parents=True, exist_ok=True)

        state.metadata["thread_id"] = thread_id
        state.metadata["thread_data"] = {
            "workspace_dir": str(workspace),
            "uploads_dir": str(uploads),
            "outputs_dir": str(outputs),
        }

        # Inject workspace path info into system prompt
        if state.messages and state.messages[0].get("role") == "system":
            workspace_info = (
                f"\n\n## Workspace\n"
                f"- Workspace: {workspace}\n"
                f"- Uploads: {uploads}\n"
                f"- Outputs: {outputs}\n"
            )
            state.messages[0]["content"] += workspace_info

        return None

    @staticmethod
    def ensure_dirs_exist(thread_data: dict) -> None:
        """Create thread directories if they don't exist (lazy init)."""
        for key in ("workspace_dir", "uploads_dir", "outputs_dir"):
            path = thread_data.get(key)
            if path and not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

    def __repr__(self) -> str:
        return f"ThreadDataMiddleware(base_dir={self._base}, lazy_init={self._lazy_init})"
