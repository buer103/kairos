"""Kairos Plugin — Memory Backend Plugin.

Provides: SQLite + FTS5 persistent memory for Kairos agents.
Registers as a 'memory' type plugin with automatic backend discovery.
"""

from __future__ import annotations

PLUGIN_NAME = "kairos-memory-backend"
PLUGIN_VERSION = "1.0.0"
PLUGIN_TYPE = "memory"
PLUGIN_DESCRIPTION = "SQLite + FTS5 persistent memory backend for Kairos agents"


def register(manager) -> None:
    """Register the memory backend plugin.

    Called by PluginManager on load. Injects the SQLiteBackend as a
    discoverable memory backend into the Kairos memory module.

    Args:
        manager: The PluginManager instance
    """
    from kairos.memory.backends import SQLiteBackend, DictBackend

    manager._registry["memory"]["sqlite"] = {
        "class": SQLiteBackend,
        "description": "SQLite + FTS5 full-text search memory backend",
        "version": PLUGIN_VERSION,
    }
    manager._registry["memory"]["dict"] = {
        "class": DictBackend,
        "description": "In-memory dict-based memory backend (lightweight)",
        "version": PLUGIN_VERSION,
    }


def get_backend(db_path: str = "~/.kairos/memory.db") -> object:
    """Convenience: instantiate the default memory backend."""
    from kairos.memory.backends import SQLiteBackend
    return SQLiteBackend(db_path=db_path)
