"""Tests for VectorMemoryBackend and MemoryRouter."""

import pytest
import sys

from kairos.memory import VectorMemoryBackend, MemoryRouter, DictBackend, SQLiteBackend


# ── VectorMemoryBackend tests ──────────────────────────────────────

@pytest.mark.skipif(
    "chromadb" not in sys.modules,
    reason="chromadb not installed",
)
class TestVectorMemoryBackend:
    """Integration tests requiring chromadb."""

    @pytest.fixture
    def backend(self, tmp_path):
        return VectorMemoryBackend(persist_dir=tmp_path / "chroma")

    def test_save_and_load(self, backend):
        backend.save("key1", "hello world", category="test")
        result = backend.load("key1")
        assert result is not None
        assert result["key"] == "key1"
        assert result["value"] == "hello world"
        assert result["category"] == "test"

    def test_load_missing(self, backend):
        assert backend.load("nonexistent") is None

    def test_delete(self, backend):
        backend.save("key1", "value1")
        assert backend.delete("key1") is True
        assert backend.load("key1") is None

    def test_delete_missing(self, backend):
        assert backend.delete("nonexistent") is False

    def test_search_semantic(self, backend):
        backend.save("pref_lang", "User prefers Chinese language")
        backend.save("pref_style", "User prefers concise responses")
        backend.save("fact_project", "Project uses pytest for testing")

        results = backend.search("language preference")
        assert len(results) > 0
        # The language entry should rank highest
        assert results[0]["key"] == "pref_lang"

    def test_search_category_filter(self, backend):
        backend.save("k1", "Chinese", category="language")
        backend.save("k2", "English", category="language")
        backend.save("k3", "Python", category="skill")

        results = backend.search("language", category="language")
        assert all(r["category"] == "language" for r in results)

    def test_upsert(self, backend):
        backend.save("key1", "old value", category="old")
        backend.save("key1", "new value", category="new")
        result = backend.load("key1")
        assert result["value"] == "new value"
        assert result["category"] == "new"

    def test_metadata(self, backend):
        backend.save("k1", "v1", metadata={"confidence": 0.9, "source": "test"})
        result = backend.load("k1")
        assert result["metadata"]["confidence"] == 0.9
        assert result["metadata"]["source"] == "test"

    def test_ttl_pruning(self, backend):
        # Save with short TTL
        backend.save("k1", "v1", ttl=0.01)
        import time
        time.sleep(0.05)
        # Should be pruned
        result = backend.load("k1")
        assert result is None

    def test_count(self, backend):
        assert backend.count() == 0
        backend.save("k1", "v1")
        backend.save("k2", "v2")
        assert backend.count() == 2

    def test_clear(self, backend):
        backend.save("k1", "v1")
        backend.save("k2", "v2")
        count = backend.clear()
        assert count == 2
        assert backend.count() == 0

    def test_search_limit(self, backend):
        for i in range(10):
            backend.save(f"k{i}", f"value about memory {i}", category="test")
        results = backend.search("memory", limit=5)
        assert len(results) <= 5

    def test_repr(self, backend):
        r = repr(backend)
        assert "VectorMemoryBackend" in r
        assert "chroma" in r


# ── MemoryRouter tests ─────────────────────────────────────────────

class TestMemoryRouter:
    """Tests that don't require chromadb."""

    @pytest.fixture
    def keyword(self):
        return DictBackend()

    @pytest.fixture
    def router(self, keyword):
        return MemoryRouter(keyword_backend=keyword)

    def test_save_to_keyword(self, router, keyword):
        router.save("k1", "hello world", category="test")
        result = keyword.load("k1")
        assert result is not None
        assert result["value"] == "hello world"

    def test_load_from_keyword(self, router):
        router.save("k1", "value1")
        result = router.load("k1")
        assert result["value"] == "value1"

    def test_delete(self, router):
        router.save("k1", "value1")
        assert router.delete("k1") is True
        assert router.load("k1") is None

    def test_search_keyword_only(self, router):
        router.save("pref_lang", "Chinese", category="preference")
        router.save("pref_style", "concise", category="preference")
        results = router.search("Chinese")
        assert len(results) > 0
        assert any("Chinese" in r.get("value", "") for r in results)

    def test_search_routes_to_keyword_for_long_query(self, router):
        router.save("k1", "value1")
        results = router.search("this is a long query with many words", limit=5)
        assert isinstance(results, list)

    def test_auto_strategy_routes_short_to_merge(self, router):
        # Short query with only keyword backend → still keyword_only
        router.save("k1", "hello world")
        results = router.search("hello")
        assert isinstance(results, list)

    def test_save_with_metadata(self, router, keyword):
        router.save("k1", "v1", metadata={"confidence": 0.8})
        result = keyword.load("k1")
        # DictBackend metadata is JSON string
        meta = result["metadata"]
        if isinstance(meta, str):
            import json
            meta = json.loads(meta)
        assert meta["confidence"] == 0.8

    def test_repr(self, router):
        r = repr(router)
        assert "MemoryRouter" in r

    def test_count(self, router):
        router.save("k1", "v1")
        counts = router.count()
        assert "keyword" in counts

    def test_clear(self, router, keyword):
        router.save("k1", "v1")
        assert router.load("k1") is not None
        router.clear()
        assert router.load("k1") is None


class TestMemoryRouterWithVector:
    """Tests with both keyword and vector backends."""

    @pytest.fixture
    def dual_router(self, tmp_path):
        keyword = DictBackend()
        try:
            vector = VectorMemoryBackend(persist_dir=tmp_path / "chroma")
        except (ImportError, ModuleNotFoundError):
            pytest.skip("chromadb not available")
        return MemoryRouter(
            keyword_backend=keyword,
            vector_backend=vector,
            search_strategy="merge",
        )

    def test_save_to_both_backends(self, dual_router):
        dual_router.save("k1", "hello world", category="test")
        # Should be in both
        kw_result = dual_router.keyword_backend.load("k1")
        vec_result = dual_router.vector_backend.load("k1")
        assert kw_result is not None
        assert vec_result is not None

    def test_merge_results(self, dual_router):
        dual_router.save("pref_lang", "User prefers Chinese", category="preference")
        dual_router.save("pref_style", "User prefers concise", category="preference")
        dual_router.save("fact_project", "Project uses pytest", category="fact")

        results = dual_router.search("language preference", limit=10)
        assert len(results) > 0
        # Results should have _source marker
        assert any("_source" in r for r in results)


# ── import guard tests ─────────────────────────────────────────────

class TestImportGuard:
    """VectorMemoryBackend import behavior without chromadb."""

    def test_raises_import_error_when_chromadb_missing(self, monkeypatch):
        """When chromadb is not installed, instantiation should raise ImportError."""
        # Force the module-level check to return False
        import kairos.memory.vector_backend as vb

        monkeypatch.setattr(vb, "_IS_AVAILABLE", False)
        with pytest.raises(ImportError, match="chromadb"):
            vb.VectorMemoryBackend()
