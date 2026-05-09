"""Kairos Plugin — Context Compression Plugin.

Provides advanced context compression strategies for Kairos agents.
Registers as a 'middleware' type plugin.
"""

from __future__ import annotations

PLUGIN_NAME = "kairos-context-compression"
PLUGIN_VERSION = "1.0.0"
PLUGIN_TYPE = "middleware"
PLUGIN_DESCRIPTION = "Advanced context compression with trajectory summarization and importance scoring"


def register(manager) -> None:
    """Register the context compression middleware plugin.

    Makes the TrajectoryCompressor and ImportanceScorer available as
    pluggable middleware layers through the PluginManager.

    Args:
        manager: The PluginManager instance
    """
    from kairos.middleware.trajectory_compressor import TrajectoryCompressor
    from kairos.middleware.importance_scorer import ImportanceScorer

    manager._registry["middleware"]["trajectory_compressor"] = {
        "class": TrajectoryCompressor,
        "description": "LLM-powered message summarization with keep-recent-N strategy",
        "version": PLUGIN_VERSION,
        "position": "before_model",
    }
    manager._registry["middleware"]["importance_scorer"] = {
        "class": ImportanceScorer,
        "description": "Per-message retention scoring with greedy token budget allocation",
        "version": PLUGIN_VERSION,
        "position": "before_model",
    }


def create_compressor(
    max_tokens: int = 8000,
    keep_recent: int = 3,
) -> object:
    """Convenience: instantiate a TrajectoryCompressor."""
    from kairos.middleware.trajectory_compressor import TrajectoryCompressor
    return TrajectoryCompressor(max_tokens=max_tokens, keep_recent=keep_recent)
