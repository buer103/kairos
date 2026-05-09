"""Subagent limit — caps concurrent delegate_task calls with dynamic limits.

DeerFlow layer 13. Per-tool limits + total budget enforcement.
"""

from __future__ import annotations

import logging
from typing import Any

from kairos.core.middleware import Middleware

logger = logging.getLogger("kairos.middleware.subagent_limit")


class SubagentLimitMiddleware(Middleware):
    """Hard cap on concurrent sub-agent tool calls.

    Hook: wrap_model_call — truncates excess calls before execution.

    Configurable per-tool limits and total budget.
    """

    TOOLS = ("task", "delegate_task", "delegate")

    def __init__(
        self,
        max_concurrent: int = 3,
        max_total_tools: int = 10,
        per_tool_limits: dict[str, int] | None = None,
    ):
        self.max_concurrent = max(1, max_concurrent)
        self.max_total = max(1, max_total_tools)
        self._per_tool = per_tool_limits or {}
        self._truncated_count = 0

    def wrap_model_call(self, messages: list[dict], handler, **kwargs) -> Any:
        response = handler(messages, **kwargs)

        if not hasattr(response, "choices") or not response.choices:
            return response

        msg = response.choices[0].message
        if not msg.tool_calls:
            return response

        all_calls = list(msg.tool_calls)

        # Enforce total cap
        if len(all_calls) > self.max_total:
            logger.warning(
                "Truncating %d → %d tool calls (total limit)",
                len(all_calls), self.max_total,
            )
            all_calls = all_calls[:self.max_total]
            self._truncated_count += len(msg.tool_calls) - self.max_total

        # Enforce per-tool caps
        tool_counts: dict[str, int] = {}
        kept = []
        for tc in all_calls:
            name = tc.function.name
            limit = self._per_tool.get(name, self.max_concurrent)
            count = tool_counts.get(name, 0)
            if count < limit:
                kept.append(tc)
                tool_counts[name] = count + 1
            else:
                self._truncated_count += 1
                logger.debug("Dropped excess '%s' call (limit=%d)", name, limit)

        if len(kept) < len(all_calls):
            logger.info(
                "Tool call enforcement: %d → %d (%d dropped)",
                len(all_calls), len(kept), len(all_calls) - len(kept),
            )

        msg.tool_calls = kept
        return response

    @property
    def truncated_count(self) -> int:
        return self._truncated_count

    def __repr__(self) -> str:
        return f"SubagentLimitMiddleware(max={self.max_concurrent}, total={self.max_total})"
