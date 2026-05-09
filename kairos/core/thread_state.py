"""Thread data types — TypedDict for type-safe middleware state.

Follows DeerFlow's ThreadDataState pattern with NotRequired fields.
"""

from __future__ import annotations

from typing import TypedDict


class ThreadDataState(TypedDict, total=False):
    """Type-safe thread data injected by ThreadDataMiddleware.

    Fields:
        workspace: Main working directory for tool execution.
        uploads: Directory for user-uploaded files.
        outputs: Directory for tool-produced output files.
        thread_root: Root directory for this thread's user data.
        thread_id: The thread/session identifier.
        user_id: The user identifier (if per-user isolation is enabled).
        run_id: The current run identifier.
    """

    workspace: str
    uploads: str
    outputs: str
    thread_root: str
    thread_id: str
    user_id: str | None
    run_id: str
