"""Tests for session persistence backends: FileSessionBackend and RedisSessionBackend.

Tests StatefulAgent integration with both backends.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from kairos.core.session_backends import (
    FileSessionBackend,
    RedisSessionBackend,
    SessionBackend,
)
from kairos.core.stateful_agent import StatefulAgent


# ============================================================================
# Test fixtures
# ============================================================================


def _make_session_data(name: str = "test-session", session_id: str = "abc123") -> dict:
    return {
        "session_id": session_id,
        "name": name,
        "saved_at": 1715000000.0,
        "turn_count": 3,
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ],
        "metadata": {"agent": "kairos", "version": "0.15.0"},
    }


# ============================================================================
# FileSessionBackend
# ============================================================================


class TestFileSessionBackend:
    @pytest.fixture
    def backend(self, tmp_path):
        return FileSessionBackend(directory=tmp_path / "sessions")

    def test_save_and_load(self, backend):
        data = _make_session_data()
        backend.save("abc123", "my-chat", data)

        loaded = backend.load("my-chat")
        assert loaded is not None
        assert loaded["session_id"] == "abc123"
        assert loaded["turn_count"] == 3
        assert len(loaded["messages"]) == 3

    def test_load_missing(self, backend):
        assert backend.load("nonexistent") is None

    def test_list_sessions(self, backend):
        backend.save("id1", "chat1", _make_session_data("chat1", "id1"))
        backend.save("id2", "chat2", _make_session_data("chat2", "id2"))

        sessions = backend.list_sessions()
        assert len(sessions) == 2
        names = {s["name"] for s in sessions}
        assert names == {"chat1", "chat2"}

    def test_delete(self, backend):
        backend.save("id1", "chat1", _make_session_data())
        assert backend.delete("chat1") is True
        assert backend.load("chat1") is None
        assert backend.delete("nonexistent") is False

    def test_overwrite(self, backend):
        data1 = _make_session_data("chat", "id1")
        backend.save("id1", "chat", data1)

        data2 = _make_session_data("chat", "id2")
        data2["turn_count"] = 10
        backend.save("id2", "chat", data2)

        loaded = backend.load("chat")
        assert loaded["session_id"] == "id2"
        assert loaded["turn_count"] == 10

    def test_directory_created(self, tmp_path):
        d = tmp_path / "nested" / "sessions"
        backend = FileSessionBackend(directory=d)
        assert d.exists()
        assert d.is_dir()


# ============================================================================
# RedisSessionBackend (unit tests — mock Redis)
# ============================================================================


class TestRedisSessionBackend:
    @pytest.fixture
    def backend(self):
        backend = RedisSessionBackend(redis_url="redis://localhost:6379/0")
        # Inject mock Redis client
        backend._redis = _MockRedis()
        return backend

    def test_save_and_load(self, backend):
        data = _make_session_data()
        backend.save("abc123", "my-chat", data)

        loaded = backend.load("my-chat")
        assert loaded is not None
        assert loaded["session_id"] == "abc123"
        assert loaded["turn_count"] == 3

    def test_load_missing(self, backend):
        assert backend.load("nonexistent") is None

    def test_list_sessions(self, backend):
        backend.save("id1", "chat1", _make_session_data("chat1", "id1"))
        backend.save("id2", "chat2", _make_session_data("chat2", "id2"))

        sessions = backend.list_sessions()
        assert len(sessions) >= 2
        names = {s["name"] for s in sessions}
        assert "chat1" in names
        assert "chat2" in names

    def test_delete(self, backend):
        backend.save("id1", "chat1", _make_session_data())
        assert backend.delete("chat1") is True
        assert backend.load("chat1") is None

    def test_flush(self, backend):
        backend.save("id1", "chat1", _make_session_data("chat1", "id1"))
        backend.save("id2", "chat2", _make_session_data("chat2", "id2"))
        count = backend.flush()
        assert count > 0
        assert backend.load("chat1") is None
        assert backend.load("chat2") is None

    def test_ping_with_mock(self, backend):
        backend._redis._ping = True
        assert backend.ping() is True

    def test_ping_failure(self, backend):
        backend._redis._ping = False
        assert backend.ping() is False

    def test_lazy_client_init(self):
        """Client should not be imported at construction time."""
        backend = RedisSessionBackend(redis_url="redis://localhost:6379/0")
        assert backend._redis is None  # not initialized yet

    def test_key_prefix(self, backend):
        backend.save("abc", "test", _make_session_data(name="test", session_id="abc"))
        # Verify key format
        assert backend._key("test") == "kairos:session:test"
        assert backend._index_key() == "kairos:sessions"


# ============================================================================
# Mock Redis
# ============================================================================


class _MockRedis:
    """Minimal Redis mock for testing RedisSessionBackend without a real server."""

    def __init__(self):
        self._store: dict[str, str] = {}
        self._sets: dict[str, set] = {}

    def get(self, key: str) -> str | None:
        return self._store.get(key)

    def set(self, key: str, value: str, *args, **kwargs) -> bool:
        self._store[key] = value
        return True

    def delete(self, *keys: str) -> int:
        count = 0
        for k in keys:
            if k in self._store:
                del self._store[k]
                count += 1
            if k in self._sets:
                del self._sets[k]
        return count

    def smembers(self, key: str) -> set:
        return self._sets.get(key, set())

    def sadd(self, key: str, *values: str) -> int:
        if key not in self._sets:
            self._sets[key] = set()
        added = 0
        for v in values:
            if v not in self._sets[key]:
                self._sets[key].add(v)
                added += 1
        return added

    def srem(self, key: str, *values: str) -> int:
        if key not in self._sets:
            return 0
        removed = 0
        for v in values:
            if v in self._sets[key]:
                self._sets[key].discard(v)
                removed += 1
        return removed

    def expire(self, key: str, ttl: int) -> bool:
        return True  # no-op in mock

    def pipeline(self):
        return _MockPipeline(self)

    def ping(self) -> bool:
        return self._ping if hasattr(self, "_ping") else True


class _MockPipeline:
    def __init__(self, redis: _MockRedis):
        self._redis = redis
        self._commands: list[tuple] = []

    def set(self, key: str, value: str, *args, **kwargs):
        self._commands.append(("set", key, value, args, kwargs))
        return self

    def expire(self, key: str, ttl: int):
        self._commands.append(("expire", key, ttl))
        return self

    def sadd(self, key: str, *values: str):
        self._commands.append(("sadd", key, values))
        return self

    def delete(self, *keys: str):
        self._commands.append(("delete", keys))
        return self

    def srem(self, key: str, *values: str):
        self._commands.append(("srem", key, values))
        return self

    def execute(self) -> list:
        results = []
        for cmd in self._commands:
            op = cmd[0]
            if op == "set":
                self._redis.set(cmd[1], cmd[2])
                results.append(True)
            elif op == "expire":
                results.append(True)
            elif op == "sadd":
                results.append(self._redis.sadd(cmd[1], *cmd[2]))
            elif op == "delete":
                results.append(self._redis.delete(*cmd[1]))
            elif op == "srem":
                results.append(self._redis.srem(cmd[1], *cmd[2]))
            else:
                results.append(None)
        return results


# ============================================================================
# StatefulAgent with custom backend
# ============================================================================


class TestStatefulAgentBackend:
    FAKE_KEY = "sk-" + "t" * 20  # avoid ContentRedactor

    @pytest.fixture
    def model_config(self):
        from kairos.providers.base import ModelConfig
        return ModelConfig(api_key=self.FAKE_KEY)

    def test_default_backend_is_file(self, model_config):
        agent = StatefulAgent(model=model_config)
        assert isinstance(agent.session_backend, FileSessionBackend)

    def test_custom_backend(self, model_config):
        backend = FileSessionBackend(directory=tempfile.mkdtemp())
        agent = StatefulAgent(model=model_config, session_backend=backend)
        assert agent.session_backend is backend

    def test_save_load_roundtrip(self, model_config):
        backend = FileSessionBackend(directory=tempfile.mkdtemp())
        agent = StatefulAgent(model=model_config, session_backend=backend)

        state = agent._init_conversation("Hello")
        agent._state = state
        agent._auto_save = False

        agent.save_session("my-chat")
        loaded = backend.load("my-chat")
        assert loaded is not None
        assert loaded["session_id"] == agent.session_id

        agent.reset()
        assert agent.load_session("my-chat") is True
        assert len(agent.history) >= 1  # has user message (assistant not yet called)

    def test_save_raises_without_active_conversation(self, model_config):
        agent = StatefulAgent(model=model_config)
        with pytest.raises(ValueError, match="No active conversation"):
            agent.save_session("empty")

    def test_load_nonexistent(self, model_config):
        agent = StatefulAgent(model=model_config)
        assert agent.load_session("nonexistent") is False
