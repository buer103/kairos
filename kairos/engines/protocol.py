"""Context Engine plugin protocol — pluggable context compression strategies.

Users implement this protocol to provide custom compression algorithms.
Default: ContextCompressor v3 (TrajectoryCompressor + ImportanceScorer + LLM summarization).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from kairos.core.state import ThreadState


class ContextEngine(ABC):
    """Protocol for pluggable context compression engines.

    Implement this to replace the default ContextCompressor with your own
    strategy — e.g., sliding window, semantic chunking, external summarizer.

    Usage::

        class MyEngine(ContextEngine):
            def compress(self, state, runtime, budget_ratio):
                # custom compression logic
                return compressed_messages, stats

        agent = Agent.build_default(
            model=model,
            context_engine=MyEngine(),
        )
    """

    @abstractmethod
    def compress(
        self,
        state: ThreadState,
        runtime: dict[str, Any],
        budget_ratio: float = 0.8,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Compress the message history to fit within a token budget.

        Args:
            state: Current thread state with full message history.
            runtime: Runtime context dict (model info, token counter, etc.).
            budget_ratio: Target fraction of max_tokens to fit within (0-1).

        Returns:
            Tuple of (compressed_messages, stats_dict).
            stats_dict should include at minimum: {"engine": self.name, "before": N, "after": M}.
        """
        ...

    @property
    def name(self) -> str:
        """Human-readable engine name for logging/stats."""
        return self.__class__.__name__

    def on_compress_start(self, state: ThreadState, runtime: dict[str, Any]) -> None:
        """Hook called before compression begins. Override for setup."""
        pass

    def on_compress_end(self, state: ThreadState, runtime: dict[str, Any], stats: dict[str, Any]) -> None:
        """Hook called after compression completes. Override for cleanup."""
        pass
