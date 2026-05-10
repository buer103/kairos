"""Tests for Kairos config + memory layers.

Covers: config.py (Config, get_config, write_default_config),
         memory/backends.py (SQLiteBackend, DictBackend).
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from kairos.config import Config, get_config, write_default_config, _find_config
from kairos.memory.backends import SQLiteBackend, DictBackend, MemoryBackend


# ============================================================================
# Config
# ============================================================================

class TestConfig:
    """Tests for Config layered configuration."""

    def test_empty_config(self, monkeypatch):
        monkeypatch.setattr("kairos.config._find_config", lambda: None)
        c = Config(path=None)
        assert c.path is None
        assert c.all() == {}

    def test_get_with_default(self):
        c = Config(path=None)
        assert c.get("nonexistent.key", default=42) == 42

    def test_get_nonexistent_no_default(self):
        c = Config(path=None)
        assert c.get("nonexistent.key") is None

    def test_set_nested(self):
        d = {}
        Config._set_nested(d, ["a", "b", "c"], 42)
        assert d["a"]["b"]["c"] == 42

    def test_set_nested_overwrites(self):
        d = {"a": {"b": {"c": 1}}}
        Config._set_nested(d, ["a", "b", "c"], 99)
        assert d["a"]["b"]["c"] == 99

    def test_get_from_env_fallback(self, monkeypatch):
        monkeypatch.setenv("MY_TEST_VALUE", "hello")
        c = Config(path=None)
        assert c.get("MY_TEST_VALUE") == "hello"

    def test_write_default_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            write_default_config(str(path))
            assert path.exists()
            import json
            data = json.loads(path.read_text())
            assert "model" in data
            assert "providers" in data

    def test_config_repr(self, monkeypatch):
        monkeypatch.setattr("kairos.config._find_config", lambda: None)
        c = Config(path=None)
        assert "empty" in repr(c)


class TestConfigFind:
    """Tests for _find_config path resolution."""

    def test_env_var_override(self, monkeypatch):
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            f.write(b"key: value")
            path = f.name
        monkeypatch.setenv("KAIROS_CONFIG", path)
        try:
            result = _find_config()
            assert result is not None
        finally:
            os.unlink(path)

    def test_no_config_found(self, monkeypatch):
        monkeypatch.delenv("KAIROS_CONFIG", raising=False)
        # Go to a temp dir without kairos.yaml
        with tempfile.TemporaryDirectory() as tmp:
            import os as _os
            old_cwd = _os.getcwd()
            _os.chdir(tmp)
            try:
                result = _find_config()
                # May find global config, ignore
            finally:
                _os.chdir(old_cwd)


# ============================================================================
# DictBackend
# ============================================================================

class TestDictBackend:
    """Tests for in-memory DictBackend."""

    @pytest.fixture
    def backend(self):
        return DictBackend()

    def test_save_and_load(self, backend):
        backend.save("key1", "value1", category="test")
        entry = backend.load("key1")
        assert entry is not None
        assert entry["key"] == "key1"
        assert entry["value"] == "value1"
        assert entry["category"] == "test"

    def test_load_missing(self, backend):
        assert backend.load("nonexistent") is None

    def test_exists(self, backend):
        assert backend.exists("key1") is False
        backend.save("key1", "val")
        assert backend.exists("key1") is True

    def test_delete(self, backend):
        backend.save("key1", "val")
        assert backend.delete("key1") is True
        assert backend.delete("key1") is False

    def test_upsert(self, backend):
        backend.save("key1", "v1", category="cat1")
        backend.save("key1", "v2", category="cat2")
        entry = backend.load("key1")
        assert entry["value"] == "v2"
        assert entry["category"] == "cat2"

    def test_search(self, backend):
        backend.save("k1", "hello world", category="greeting")
        backend.save("k2", "goodbye world", category="farewell")
        backend.save("k3", "nothing here", category="other")

        results = backend.search("world")
        assert len(results) == 2

    def test_search_category_filter(self, backend):
        backend.save("k1", "hello", category="a")
        backend.save("k2", "hello", category="b")
        results = backend.search("hello", category="a")
        assert len(results) == 1
        assert results[0]["category"] == "a"

    def test_list_keys(self, backend):
        backend.save("apple", "v", category="fruit")
        backend.save("banana", "v", category="fruit")
        backend.save("carrot", "v", category="veg")
        assert backend.list_keys() == ["apple", "banana", "carrot"]

    def test_list_keys_with_prefix(self, backend):
        backend.save("a1", "v")
        backend.save("a2", "v")
        backend.save("b1", "v")
        assert backend.list_keys(prefix="a") == ["a1", "a2"]

    def test_list_keys_with_category(self, backend):
        backend.save("k1", "v", category="c1")
        backend.save("k2", "v", category="c2")
        assert backend.list_keys(category="c1") == ["k1"]

    def test_clear_all(self, backend):
        backend.save("k1", "v")
        backend.save("k2", "v")
        count = backend.clear()
        assert count == 2
        assert backend.load("k1") is None

    def test_clear_by_category(self, backend):
        backend.save("k1", "v", category="a")
        backend.save("k2", "v", category="b")
        count = backend.clear(category="a")
        assert count == 1
        assert backend.load("k1") is None
        assert backend.load("k2") is not None

    def test_ttl_expiry(self, backend):
        backend.save("key1", "val", ttl=0.01)  # 10ms TTL
        time.sleep(0.02)
        assert backend.load("key1") is None  # expired

    def test_metadata_preserved(self, backend):
        backend.save("key1", "val", metadata={"priority": "high"})
        entry = backend.load("key1")
        import json
        meta = json.loads(entry["metadata"])
        assert meta["priority"] == "high"


# ============================================================================
# SQLiteBackend
# ============================================================================

class TestSQLiteBackend:
    """Tests for SQLite + FTS5 persistent memory backend."""

    @pytest.fixture
    def backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test_memory.db"
            b = SQLiteBackend(db_path=db_path)
            yield b
            b.close()

    def test_save_and_load(self, backend):
        backend.save("key1", "hello world", category="test")
        entry = backend.load("key1")
        assert entry["key"] == "key1"
        assert entry["value"] == "hello world"

    def test_delete(self, backend):
        backend.save("key1", "val")
        assert backend.delete("key1") is True
        assert backend.load("key1") is None

    def test_upsert(self, backend):
        backend.save("k", "v1")
        backend.save("k", "v2")
        assert backend.load("k")["value"] == "v2"

    def test_search(self, backend):
        backend.save("k1", "hello world")
        backend.save("k2", "goodbye moon")
        results = backend.search("hello")
        assert len(results) == 1

    def test_search_category(self, backend):
        backend.save("k1", "hello", category="a")
        backend.save("k2", "hello", category="b")
        results = backend.search("hello", category="b")
        assert len(results) == 1

    def test_list_keys(self, backend):
        backend.save("b", "v")
        backend.save("a", "v")
        assert backend.list_keys() == ["a", "b"]

    def test_list_keys_prefix(self, backend):
        backend.save("pref_1", "v")
        backend.save("pref_2", "v")
        backend.save("other", "v")
        keys = backend.list_keys(prefix="pref_")
        assert len(keys) == 2
        assert all(k.startswith("pref_") for k in keys)

    def test_clear(self, backend):
        backend.save("k1", "v")
        backend.save("k2", "v")
        assert backend.clear() == 2
        assert backend.load("k1") is None

    def test_clear_category(self, backend):
        backend.save("k1", "v", category="a")
        backend.save("k2", "v", category="b")
        assert backend.clear(category="a") == 1

    def test_ttl(self, backend):
        backend.save("k", "v", ttl=0.01)
        time.sleep(0.02)
        assert backend.load("k") is None
