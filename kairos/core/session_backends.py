"""Session persistence backends — pluggable storage for conversation state.

Backends:
    FileSessionBackend  — local JSON files (default, zero-dependency)
    RedisSessionBackend — Redis key-value store (production, multi-process safe)

Protocol:
    All backends implement save/load/list/delete with the same interface.
    StatefulAgent delegates persistence to a backend instance.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


# ============================================================================
# Protocol
# ============================================================================


@runtime_checkable
class SessionBackend(Protocol):
    """Protocol for session persistence backends.

    Any object implementing save/load/list/delete with these signatures
    is a valid SessionBackend — no need to inherit.
    """

    def save(self, session_id: str, name: str, data: dict[str, Any]) -> None:
        """Persist session data keyed by name."""
        ...

    def load(self, name: str) -> dict[str, Any] | None:
        """Load session data by name. Returns None if not found."""
        ...

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all saved sessions with metadata."""
        ...

    def delete(self, name: str) -> bool:
        """Delete a session by name. Returns True if deleted."""
        ...


class AbstractSessionBackend(ABC):
    """Optional ABC base class with JSON serialize/deserialize helpers."""

    @staticmethod
    def _serialize(data: dict[str, Any]) -> str:
        return json.dumps(data, ensure_ascii=False, indent=2, default=str)

    @staticmethod
    def _deserialize(raw: str) -> dict[str, Any]:
        return json.loads(raw)

    @abstractmethod
    def save(self, session_id: str, name: str, data: dict[str, Any]) -> None: ...
    @abstractmethod
    def load(self, name: str) -> dict[str, Any] | None: ...
    @abstractmethod
    def list_sessions(self) -> list[dict[str, Any]]: ...
    @abstractmethod
    def delete(self, name: str) -> bool: ...


# ============================================================================
# File Backend (default)
# ============================================================================


class FileSessionBackend:
    """Local JSON-file session persistence.

    Sessions stored as ~/.kairos/sessions/<name>.json
    """

    def __init__(self, directory: str | Path = "~/.kairos/sessions"):
        self._dir = Path(directory).expanduser().resolve()
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def directory(self) -> Path:
        return self._dir

    def save(self, session_id: str, name: str, data: dict[str, Any]) -> None:
        path = self._dir / f"{name}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str))

    def load(self, name: str) -> dict[str, Any] | None:
        path = self._dir / f"{name}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def list_sessions(self) -> list[dict[str, Any]]:
        sessions = []
        for f in sorted(self._dir.glob("*.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                d = json.loads(f.read_text())
                sessions.append({
                    "name": d.get("name", f.stem),
                    "session_id": d.get("session_id"),
                    "saved_at": d.get("saved_at"),
                    "turn_count": d.get("turn_count", 0),
                    "message_count": len(d.get("messages", [])),
                })
            except Exception:
                pass
        return sessions

    def delete(self, name: str) -> bool:
        path = self._dir / f"{name}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    def rename(self, old_name: str, new_name: str) -> bool:
        old_path = self._dir / f"{old_name}.json"
        new_path = self._dir / f"{new_name}.json"
        if not old_path.exists():
            return False
        old_path.rename(new_path)
        return True


# ============================================================================
# Redis Backend (production)
# ============================================================================


class RedisSessionBackend:
    """Redis-backed session persistence for multi-process deployments.

    Sessions stored as Redis keys: kairos:session:<name>
    Index stored as Redis set: kairos:sessions

    Usage:
        backend = RedisSessionBackend(redis_url="redis://localhost:6379/0")
        agent = StatefulAgent(session_backend=backend)
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "kairos:session",
        ttl: int | None = None,  # seconds, None = no expiry
    ):
        self._redis_url = redis_url
        self._key_prefix = key_prefix
        self._ttl = ttl
        self._redis = None  # lazy init

    @property
    def client(self):
        """Lazy Redis client — import only when needed."""
        if self._redis is None:
            import redis as _redis
            self._redis = _redis.from_url(self._redis_url, decode_responses=True)
        return self._redis

    def _key(self, name: str) -> str:
        return f"{self._key_prefix}:{name}"

    def _index_key(self) -> str:
        return f"{self._key_prefix}s"

    def save(self, session_id: str, name: str, data: dict[str, Any]) -> None:
        key = self._key(name)
        payload = json.dumps(data, ensure_ascii=False, default=str)
        pipe = self.client.pipeline()
        pipe.set(key, payload)
        if self._ttl:
            pipe.expire(key, self._ttl)
        pipe.sadd(self._index_key(), name)
        pipe.execute()

    def load(self, name: str) -> dict[str, Any] | None:
        raw = self.client.get(self._key(name))
        if raw is None:
            return None
        return json.loads(raw)

    def list_sessions(self) -> list[dict[str, Any]]:
        names = self.client.smembers(self._index_key())
        sessions = []
        for name in names:
            raw = self.client.get(self._key(name))
            if raw:
                try:
                    d = json.loads(raw)
                    sessions.append({
                        "name": d.get("name", name),
                        "session_id": d.get("session_id"),
                        "saved_at": d.get("saved_at"),
                        "turn_count": d.get("turn_count", 0),
                        "message_count": len(d.get("messages", [])),
                    })
                except Exception:
                    pass
        sessions.sort(key=lambda s: s.get("saved_at", 0), reverse=True)
        return sessions

    def delete(self, name: str) -> bool:
        pipe = self.client.pipeline()
        pipe.delete(self._key(name))
        pipe.srem(self._index_key(), name)
        results = pipe.execute()
        return results[0] > 0  # deleted key count

    def rename(self, old_name: str, new_name: str) -> bool:
        """Rename a session: copy data to new key, delete old, update index."""
        old_key = self._key(old_name)
        new_key = self._key(new_name)
        raw = self.client.get(old_key)
        if raw is None:
            return False
        pipe = self.client.pipeline()
        pipe.set(new_key, raw)
        pipe.delete(old_key)
        pipe.srem(self._index_key(), old_name)
        pipe.sadd(self._index_key(), new_name)
        pipe.execute()
        return True

    def flush(self) -> int:
        """Delete all Kairos sessions from Redis. Returns count removed."""
        names = self.client.smembers(self._index_key())
        if not names:
            return 0
        keys = [self._key(n) for n in names] + [self._index_key()]
        return self.client.delete(*keys)

    def ping(self) -> bool:
        """Check Redis connectivity."""
        try:
            return self.client.ping()
        except Exception:
            return False
