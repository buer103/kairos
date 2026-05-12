"""Memory package — persistent cross-session memory with FTS5 full-text + semantic search.

Provides:
- MemoryBackend       — abstract base class for storage engines
- SQLiteBackend       — SQLite + FTS5 with BM25 ranking (default)
- DictBackend         — in-memory dict for testing / lightweight use
- VectorMemoryBackend — ChromaDB-powered semantic search (pip install chromadb)
- MemoryRouter        — multi-backend router with auto strategy selection
- MemoryMiddleware    — injects memory context before model calls,
                        saves extracted facts after model calls
"""

from kairos.memory.backends import MemoryBackend, SQLiteBackend, DictBackend
from kairos.memory.vector_backend import VectorMemoryBackend
from kairos.memory.router import MemoryRouter
from kairos.memory.middleware import MemoryMiddleware

# Keep legacy MemoryStore accessible for backward compatibility.
from kairos.memory.store import MemoryStore  # noqa: F401

__all__ = [
    "MemoryBackend",
    "SQLiteBackend",
    "DictBackend",
    "VectorMemoryBackend",
    "MemoryRouter",
    "MemoryMiddleware",
    "MemoryStore",
]
