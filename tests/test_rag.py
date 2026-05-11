"""Tests for RAG engines — adapters + vector store."""

from __future__ import annotations

import tempfile
from pathlib import Path

from kairos.infra.rag.adapters import MarkdownAdapter, TextAdapter
from kairos.infra.rag.vector_store import VectorStore


# ============================================================================
# MarkdownAdapter
# ============================================================================


class TestMarkdownAdapter:
    """Markdown file loading with heading-based chunking."""

    def test_load_missing_file(self):
        chunks = MarkdownAdapter.load("/nonexistent/path.md")
        assert chunks == []

    def test_load_single_heading(self, tmp_path: Path):
        md = tmp_path / "test.md"
        md.write_text("# Title\nSome content here.")
        chunks = MarkdownAdapter.load(str(md))
        assert len(chunks) == 1
        assert chunks[0]["content"] == "# Title\nSome content here."
        assert chunks[0]["metadata"]["source"] == str(md)
        assert chunks[0]["metadata"]["title"] == "Title"

    def test_load_multiple_headings(self, tmp_path: Path):
        md = tmp_path / "test.md"
        md.write_text("# H1\nContent one.\n# H2\nContent two.\nSome more content.")
        chunks = MarkdownAdapter.load(str(md))
        assert len(chunks) == 2
        assert chunks[0]["metadata"]["title"] == "H1"
        assert "Content one." in chunks[0]["content"]
        assert chunks[1]["metadata"]["title"] == "H2"
        assert "Content two." in chunks[1]["content"]

    def test_load_headings_with_subheadings(self, tmp_path: Path):
        md = tmp_path / "test.md"
        md.write_text("# Main\nTop content.\n## Sub\nSub content.\n# Other\nOther content.")
        chunks = MarkdownAdapter.load(str(md))
        assert len(chunks) == 2
        assert "## Sub" in chunks[0]["content"]
        assert chunks[0]["metadata"]["title"] == "Main"

    def test_load_no_headings(self, tmp_path: Path):
        md = tmp_path / "test.md"
        md.write_text("Just some text\nwithout any headings.")
        chunks = MarkdownAdapter.load(str(md))
        assert len(chunks) == 1
        assert chunks[0]["content"] == "Just some text\nwithout any headings."
        assert chunks[0]["metadata"]["title"] == ""

    def test_load_empty_file(self, tmp_path: Path):
        md = tmp_path / "empty.md"
        md.write_text("")
        chunks = MarkdownAdapter.load(str(md))
        assert len(chunks) == 1
        assert chunks[0]["content"] == ""

    def test_load_with_path_object(self, tmp_path: Path):
        md = tmp_path / "test.md"
        md.write_text("# Title\nBody.")
        chunks = MarkdownAdapter.load(md)
        assert len(chunks) == 1


# ============================================================================
# TextAdapter
# ============================================================================


class TestTextAdapter:
    """Plain text file loading."""

    def test_load_missing_file(self):
        chunks = TextAdapter.load("/nonexistent/path.txt")
        assert chunks == []

    def test_load_simple_text(self, tmp_path: Path):
        txt = tmp_path / "test.txt"
        txt.write_text("Line 1\nLine 2")
        chunks = TextAdapter.load(str(txt))
        assert len(chunks) == 1
        assert chunks[0]["content"] == "Line 1\nLine 2"
        assert chunks[0]["metadata"]["source"] == str(txt)

    def test_load_empty_file(self, tmp_path: Path):
        txt = tmp_path / "empty.txt"
        txt.write_text("")
        chunks = TextAdapter.load(str(txt))
        assert len(chunks) == 1
        assert chunks[0]["content"] == ""


# ============================================================================
# VectorStore
# ============================================================================


class TestVectorStore:
    """In-memory vector store with keyword-based search."""

    def test_default_backend(self):
        store = VectorStore()
        assert store._backend == "memory"
        assert store.count() == 0

    def test_custom_backend(self):
        store = VectorStore(backend="chromadb")
        assert store._backend == "chromadb"

    def test_add_simple_documents(self):
        store = VectorStore()
        store.add(["hello world", "foo bar"])
        assert store.count() == 2

    def test_add_with_metadata(self):
        store = VectorStore()
        store.add(
            ["doc1", "doc2"],
            metadatas=[{"source": "a"}, {"source": "b"}],
        )
        assert store.count() == 2
        assert store._documents[0]["metadata"] == {"source": "a"}
        assert store._documents[1]["metadata"] == {"source": "b"}

    def test_add_with_ids(self):
        store = VectorStore()
        store.add(["a", "b", "c"], ids=["id1", "id2", "id3"])
        assert store._documents[0]["id"] == "id1"
        assert store._documents[1]["id"] == "id2"
        assert store._documents[2]["id"] == "id3"

    def test_add_auto_ids(self):
        store = VectorStore()
        store.add(["first", "second"])
        assert store._documents[0]["id"] == "0"
        assert store._documents[1]["id"] == "1"

    def test_search_exact_match(self):
        store = VectorStore()
        store.add(["python programming guide", "rust systems programming", "javascript web development"])
        results = store.search("python", top_k=3)
        assert len(results) >= 1
        assert results[0]["id"] == "0"
        assert "python" in results[0]["content"].lower()

    def test_search_multiple_matches(self):
        store = VectorStore()
        store.add(["python data science", "python machine learning", "rust cli tools"])
        results = store.search("python", top_k=2)
        assert len(results) == 2
        assert all("python" in r["content"].lower() for r in results)

    def test_search_no_results(self):
        store = VectorStore()
        store.add(["apple banana", "cherry date"])
        results = store.search("xyzzyy")
        assert results == []

    def test_search_partial_match(self):
        store = VectorStore()
        store.add(["hello world", "goodbye world", "nothing here"])
        results = store.search("world")
        assert len(results) == 2

    def test_search_top_k_limits(self):
        store = VectorStore()
        for i in range(10):
            store.add([f"document number {i}"])
        results = store.search("document", top_k=4)
        assert len(results) == 4

    def test_clear(self):
        store = VectorStore()
        store.add(["doc1", "doc2"])
        assert store.count() == 2
        store.clear()
        assert store.count() == 0

    def test_count(self):
        store = VectorStore()
        assert store.count() == 0
        store.add(["a"])
        assert store.count() == 1
        store.add(["b", "c"])
        assert store.count() == 3

    def test_search_returns_score(self):
        store = VectorStore()
        store.add(["exact match keyword here", "some other text"])
        results = store.search("exact match keyword")
        assert len(results) >= 1
        assert "score" in results[0]
        assert results[0]["score"] > 0

    def test_search_result_structure(self):
        store = VectorStore()
        store.add(["test content"], metadatas=[{"source": "test.txt"}], ids=["abc"])
        results = store.search("test")
        assert len(results) == 1
        r = results[0]
        assert r["id"] == "abc"
        assert r["content"] == "test content"
        assert r["metadata"] == {"source": "test.txt"}
        assert "score" in r
