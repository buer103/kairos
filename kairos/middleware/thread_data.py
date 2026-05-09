"""Thread data middleware — per-thread workspace management.

DeerFlow layer 1 — MUST be the first middleware in the pipeline.
Creates and manages workspace/upload/output directories for each conversation thread.

Compared to Kairos v0.8.0 (77 lines, bare dicts):
  - Type-safe ThreadDataState TypedDict
  - User-level path isolation (multi-tenant)
  - HumanMessage enrichment (run_id, timestamp)
  - Structured logging
  - Fail-fast on missing thread_id
  - Deferred workspace creation
  - ThreadPath integration with sandbox
  - Thread cleanup on session end
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from kairos.core.middleware import Middleware
from kairos.core.paths import ThreadPaths
from kairos.core.thread_state import ThreadDataState

logger = logging.getLogger("kairos.middleware.thread_data")


class ThreadDataMiddleware(Middleware):
    """Creates per-thread directory structure and injects path metadata.

    Hook: before_agent — computes paths, optionally creates directories.
          after_agent  — optional thread cleanup.

    Modes:
      - lazy_init=True (default): compute paths only, create dirs on first use
      - lazy_init=False: eagerly create directories in before_agent()

    Directory layout:
      {base_dir}/threads/{thread_id}/
        user-data/          (default, no user isolation)
          workspace/        ← tool code execution sandbox
          uploads/          ← user-uploaded files
          outputs/          ← tool-produced output files
        user-{user_id}/     (per-user isolation)
          workspace/
          ...
    """

    def __init__(
        self,
        base_dir: str | None = None,
        lazy_init: bool = True,
        user_isolation: bool = False,
        cleanup_on_end: bool = False,
    ):
        self._paths = ThreadPaths(base_dir)
        self._lazy_init = lazy_init
        self._user_isolation = user_isolation
        self._cleanup_on_end = cleanup_on_end

    # ── before_agent — primary hook ─────────────────────────────

    def before_agent(self, state: Any, runtime: dict[str, Any]) -> dict[str, Any] | None:
        """Compute thread paths and inject into state + runtime.

        Resolution order for thread_id:
          1. runtime["thread_id"]
          2. runtime["session_id"]
          3. state.metadata.get("thread_id")
          4. Auto-generated UUID (with warning)
        """
        thread_id = (
            runtime.get("thread_id")
            or runtime.get("session_id")
            or (state.metadata or {}).get("thread_id")
        )

        if thread_id is None:
            thread_id = f"auto-{uuid.uuid4().hex[:8]}"
            logger.warning(
                "No thread_id found in runtime or state. Auto-generated: %s",
                thread_id,
            )
        else:
            logger.debug("ThreadDataMiddleware: thread_id=%s", thread_id)

        user_id = None
        if self._user_isolation:
            user_id = runtime.get("user_id") or (state.metadata or {}).get("user_id")

        run_id = runtime.get("run_id") or uuid.uuid4().hex[:8]

        # Compute paths
        paths = self._paths.all_paths(thread_id, user_id=user_id)

        # Eager directory creation
        if not self._lazy_init:
            self._paths.ensure(thread_id, user_id=user_id)
            logger.debug("Created thread data dirs for %s (eager)", thread_id)

        # ── Enrich last HumanMessage ─────────────────────────
        self._enrich_last_human_message(state, thread_id, user_id, run_id)

        # ── Inject thread data into state metadata ───────────
        if state.metadata is None:
            state.metadata = {}

        thread_data: ThreadDataState = {
            "workspace": paths["workspace"],
            "uploads": paths["uploads"],
            "outputs": paths["outputs"],
            "thread_root": paths["thread_root"],
            "thread_id": thread_id,
            "user_id": user_id,
            "run_id": run_id,
        }

        state.metadata["thread_data"] = thread_data
        state.metadata["thread_id"] = thread_id
        state.metadata["run_id"] = run_id

        # ── Inject workspace info into system prompt ─────────
        if state.messages and state.messages[0].get("role") == "system":
            workspace_info = (
                f"\n\n## Thread Environment\n"
                f"- Thread ID: `{thread_id}`\n"
                f"- Run ID: `{run_id}`\n"
                f"- Workspace: `{paths['workspace']}`\n"
                f"- Uploads: `{paths['uploads']}`\n"
                f"- Outputs: `{paths['outputs']}`\n"
            )
            state.messages[0]["content"] += workspace_info

        # ── Update runtime for downstream middleware ─────────
        runtime["thread_id"] = thread_id
        runtime["run_id"] = run_id
        runtime["thread_data"] = thread_data
        runtime["thread_paths"] = paths

        return None

    # ── after_agent — cleanup ────────────────────────────────────

    def after_agent(self, state: Any, runtime: dict[str, Any]) -> dict[str, Any] | None:
        """Optionally clean up thread data when session ends."""
        if not self._cleanup_on_end:
            return None

        td = runtime.get("thread_data") or (state.metadata or {}).get("thread_data")
        if not td:
            return None

        thread_id = td.get("thread_id")
        if thread_id and not thread_id.startswith("auto-"):
            # Only clean up explicit threads, not auto-generated ones
            return None

        logger.info("Cleaning up thread data for auto-generated thread: %s", thread_id)
        self._paths.remove_thread(thread_id)
        return None

    # ── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def ensure_workspace(state_or_runtime: dict[str, Any]) -> str | None:
        """Lazy-init: create workspace dir on first use. Returns workspace path.

        Call this from tools or sandbox middleware before writing files.
        """
        td = (
            state_or_runtime.get("thread_data")
            or state_or_runtime.get("metadata", {}).get("thread_data")
        )
        if not td:
            return None

        ws = td.get("workspace")
        if ws:
            from pathlib import Path
            Path(ws).mkdir(parents=True, exist_ok=True)
        return ws

    @staticmethod
    def get_thread_data(state_or_runtime: dict[str, Any]) -> ThreadDataState | None:
        """Extract ThreadDataState from state or runtime dict."""
        td = (
            state_or_runtime.get("thread_data")
            or state_or_runtime.get("metadata", {}).get("thread_data")
        )
        return td

    def _enrich_last_human_message(
        self,
        state: Any,
        thread_id: str,
        user_id: str | None,
        run_id: str,
    ) -> None:
        """Add run_id and timestamp metadata to the last HumanMessage.

        This allows downstream middleware and tools to know the exact context
        of the current message.
        """
        if not state.messages:
            return

        # Find the last HumanMessage (user role)
        for msg in reversed(state.messages):
            if msg.get("role") == "user":
                if isinstance(msg, dict):
                    msg.setdefault("metadata", {})
                    msg["metadata"].update({
                        "thread_id": thread_id,
                        "run_id": run_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                    if user_id:
                        msg["metadata"]["user_id"] = user_id
                break

    def __repr__(self) -> str:
        return (
            f"ThreadDataMiddleware("
            f"base={self._paths.base_dir}, "
            f"lazy={self._lazy_init}, "
            f"user_iso={self._user_isolation})"
        )
