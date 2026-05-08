"""Evidence tracker middleware — records every tool invocation as an evidence step."""

from __future__ import annotations

import time
from typing import Any

from kairos.core.middleware import Middleware


class EvidenceTracker(Middleware):
    """Records every tool invocation as a Step in the evidence chain."""

    def wrap_tool_call(self, tool_name: str, args: dict, handler, **kwargs) -> Any:
        state = kwargs.get("state")
        start = time.time()
        result = handler(tool_name, args, **kwargs)
        elapsed = (time.time() - start) * 1000

        if state and state.case:
            step = state.case.add_step(tool_name, args)
            state.case.complete_step(step, result, elapsed)

        return result
