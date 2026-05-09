"""Thread paths — structured directory resolution for per-thread workspaces.

Follows DeerFlow's pattern with user-isolated paths and clean separation
between path computation and directory creation.

Layout:
    {base_dir}/
      threads/
        {thread_id}/
          user-data/       (default, no per-user isolation)
            workspace/     ← main working directory for tool execution
            uploads/       ← user-uploaded files
            outputs/       ← tool-produced output files
          user-{user_id}/  (per-user isolation when user_id is provided)
            workspace/
            uploads/
            outputs/
"""

from __future__ import annotations

import os
from pathlib import Path


class ThreadPaths:
    """Resolve and manage thread-specific directory paths.

    Usage:
        paths = ThreadPaths()
        ws = paths.workspace("thread-123")           # → .../threads/thread-123/user-data/workspace
        ws = paths.workspace("thread-123", "alice")  # → .../threads/thread-123/user-alice/workspace
        paths.ensure("thread-123")                   # create all dirs
    """

    def __init__(self, base_dir: str | Path | None = None):
        self._base = Path(base_dir or Path.home() / ".kairos" / "data")
        self._threads_dir = self._base / "threads"

    # ── Path computation (no disk I/O) ──────────────────────────

    def workspace(self, thread_id: str, user_id: str | None = None) -> Path:
        """Resolve the workspace directory for a thread."""
        return self._user_dir(thread_id, user_id) / "workspace"

    def uploads(self, thread_id: str, user_id: str | None = None) -> Path:
        """Resolve the uploads directory for a thread."""
        return self._user_dir(thread_id, user_id) / "uploads"

    def outputs(self, thread_id: str, user_id: str | None = None) -> Path:
        """Resolve the outputs directory for a thread."""
        return self._user_dir(thread_id, user_id) / "outputs"

    def thread_root(self, thread_id: str, user_id: str | None = None) -> Path:
        """Resolve the root directory for a thread's user data."""
        return self._user_dir(thread_id, user_id)

    def all_paths(self, thread_id: str, user_id: str | None = None) -> dict[str, str]:
        """Return all thread paths as a string-keyed dict (for injection into state)."""
        return {
            "thread_root": str(self.thread_root(thread_id, user_id)),
            "workspace": str(self.workspace(thread_id, user_id)),
            "uploads": str(self.uploads(thread_id, user_id)),
            "outputs": str(self.outputs(thread_id, user_id)),
        }

    # ── Directory creation ──────────────────────────────────────

    def ensure(self, thread_id: str, user_id: str | None = None) -> dict[str, str]:
        """Create all thread directories. Returns the path dict."""
        paths = {}
        for name in ("workspace", "uploads", "outputs"):
            p = self._user_dir(thread_id, user_id) / name
            p.mkdir(parents=True, exist_ok=True)
            paths[name] = str(p)
        return paths

    def ensure_workspace(self, thread_id: str, user_id: str | None = None) -> Path:
        """Create only the workspace directory (lazy init)."""
        p = self.workspace(thread_id, user_id)
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ── Thread cleanup ──────────────────────────────────────────

    def thread_exists(self, thread_id: str) -> bool:
        """Check if a thread directory exists on disk."""
        return (self._threads_dir / thread_id).exists()

    def list_threads(self) -> list[str]:
        """List all thread IDs with directories on disk."""
        if not self._threads_dir.exists():
            return []
        return sorted(
            d.name for d in self._threads_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    def remove_thread(self, thread_id: str) -> bool:
        """Remove all data for a thread. Returns True if anything was removed."""
        thread_dir = self._threads_dir / thread_id
        if not thread_dir.exists():
            return False
        import shutil
        shutil.rmtree(thread_dir)
        return True

    def size(self, thread_id: str) -> int:
        """Get total size in bytes of a thread's data directory."""
        thread_dir = self._threads_dir / thread_id
        if not thread_dir.exists():
            return 0
        total = 0
        for f in thread_dir.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total

    # ── Internal ─────────────────────────────────────────────────

    def _user_dir(self, thread_id: str, user_id: str | None) -> Path:
        """Resolve the per-user subdirectory within a thread."""
        if user_id:
            return self._threads_dir / thread_id / f"user-{user_id}"
        return self._threads_dir / thread_id / "user-data"

    @property
    def base_dir(self) -> Path:
        return self._base
