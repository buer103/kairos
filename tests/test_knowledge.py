"""Tests for KnowledgeSchema and KnowledgeStore."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from kairos.infra.knowledge.schema import KnowledgeSchema
from kairos.infra.knowledge.store import KnowledgeStore


# ============================================================================
# Custom Schema Subclass for testing
# ============================================================================


class FaultDiagnosis(KnowledgeSchema):
    """Test domain schema."""

    id: str
    signal_name: str
    root_cause: str
    solution: str
    severity: str = "medium"

    def __init__(
        self,
        id: str,
        signal_name: str,
        root_cause: str,
        solution: str,
        severity: str = "medium",
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
    ):
        super().__init__(id=id, created_at=created_at, updated_at=updated_at)
        self.signal_name = signal_name
        self.root_cause = root_cause
        self.solution = solution
        self.severity = severity


# ============================================================================
# KnowledgeSchema
# ============================================================================


class TestKnowledgeSchema:
    """Base schema: serialization, deserialization, repr."""

    def test_init_defaults(self):
        ks = KnowledgeSchema(id="K-001")
        assert ks.id == "K-001"
        assert isinstance(ks.created_at, datetime)
        assert isinstance(ks.updated_at, datetime)

    def test_init_explicit_timestamps(self):
        past = datetime(2024, 1, 1)
        recent = datetime(2024, 6, 1)
        ks = KnowledgeSchema(id="K-001", created_at=past, updated_at=recent)
        assert ks.created_at == past
        assert ks.updated_at == recent

    def test_to_dict(self):
        ks = KnowledgeSchema(id="K-001")
        d = ks.to_dict()
        assert d["id"] == "K-001"
        assert "created_at" in d
        assert "updated_at" in d
        # Datetimes should be ISO strings
        assert isinstance(d["created_at"], str)

    def test_from_dict_roundtrip(self):
        ks = KnowledgeSchema(id="K-001")
        d = ks.to_dict()
        restored = KnowledgeSchema.from_dict(d)
        assert restored.id == ks.id
        # from_dict stores ISO strings, not datetime objects
        assert restored.to_dict() == d

    def test_repr(self):
        ks = KnowledgeSchema(id="K-001")
        r = repr(ks)
        assert "KnowledgeSchema" in r
        assert "K-001" in r
        assert "created_at" in r


class TestKnowledgeSchemaSubclass:
    """Subclass with custom fields."""

    def test_subclass_init(self):
        fd = FaultDiagnosis(
            id="F-001",
            signal_name="engine_temp",
            root_cause="overheat",
            solution="reduce load",
        )
        assert fd.id == "F-001"
        assert fd.signal_name == "engine_temp"
        assert fd.root_cause == "overheat"
        assert fd.solution == "reduce load"
        assert fd.severity == "medium"

    def test_subclass_to_dict(self):
        fd = FaultDiagnosis(id="F-001", signal_name="s1", root_cause="r1", solution="sol1")
        d = fd.to_dict()
        assert d["id"] == "F-001"
        assert d["signal_name"] == "s1"
        assert d["root_cause"] == "r1"
        assert d["solution"] == "sol1"
        assert d["severity"] == "medium"

    def test_subclass_from_dict(self):
        d = {
            "id": "F-001",
            "signal_name": "s1",
            "root_cause": "r1",
            "solution": "sol1",
            "severity": "high",
        }
        fd = FaultDiagnosis.from_dict(d)
        assert fd.id == "F-001"
        assert fd.signal_name == "s1"
        assert fd.severity == "high"

    def test_subclass_repr(self):
        fd = FaultDiagnosis(id="F-001", signal_name="s1", root_cause="r1", solution="sol1")
        r = repr(fd)
        assert "FaultDiagnosis" in r
        assert "F-001" in r
        assert "signal_name" in r


# ============================================================================
# KnowledgeStore
# ============================================================================


class TestKnowledgeStore:
    """Structured knowledge store with typed schema."""

    def test_init(self):
        store = KnowledgeStore(schema=FaultDiagnosis)
        assert store.schema == FaultDiagnosis
        assert store.count() == 0

    def test_insert(self):
        store = KnowledgeStore(schema=FaultDiagnosis)
        fd = FaultDiagnosis(id="F-001", signal_name="s1", root_cause="r1", solution="sol1")
        store.insert(fd)
        assert store.count() == 1

    def test_insert_updates_timestamp(self):
        store = KnowledgeStore(schema=FaultDiagnosis)
        fd = FaultDiagnosis(id="F-001", signal_name="s1", root_cause="r1", solution="sol1")
        original_ts = fd.updated_at
        store.insert(fd)
        assert fd.updated_at >= original_ts

    def test_get_existing(self):
        store = KnowledgeStore(schema=FaultDiagnosis)
        fd = FaultDiagnosis(id="F-001", signal_name="s1", root_cause="r1", solution="sol1")
        store.insert(fd)
        retrieved = store.get("F-001")
        assert retrieved is not None
        assert retrieved.id == "F-001"
        assert retrieved.signal_name == "s1"

    def test_get_missing(self):
        store = KnowledgeStore(schema=FaultDiagnosis)
        assert store.get("nonexistent") is None

    def test_query_no_filter_returns_all(self):
        store = KnowledgeStore(schema=FaultDiagnosis)
        store.insert(FaultDiagnosis(id="F-001", signal_name="s1", root_cause="r1", solution="sol1"))
        store.insert(FaultDiagnosis(id="F-002", signal_name="s2", root_cause="r2", solution="sol2"))
        results = store.query()
        assert len(results) == 2

    def test_query_with_single_filter(self):
        store = KnowledgeStore(schema=FaultDiagnosis)
        store.insert(FaultDiagnosis(id="F-001", signal_name="temp", root_cause="heat", solution="cool"))
        store.insert(FaultDiagnosis(id="F-002", signal_name="pressure", root_cause="leak", solution="seal"))
        results = store.query({"root_cause": "heat"})
        assert len(results) == 1
        assert results[0].id == "F-001"

    def test_query_with_multiple_filters(self):
        store = KnowledgeStore(schema=FaultDiagnosis)
        store.insert(FaultDiagnosis(id="F-001", signal_name="temp", root_cause="heat", solution="cool", severity="high"))
        store.insert(FaultDiagnosis(id="F-002", signal_name="temp", root_cause="heat", solution="fan", severity="medium"))
        results = store.query({"signal_name": "temp", "severity": "high"})
        assert len(results) == 1
        assert results[0].id == "F-001"

    def test_query_no_match(self):
        store = KnowledgeStore(schema=FaultDiagnosis)
        store.insert(FaultDiagnosis(id="F-001", signal_name="temp", root_cause="heat", solution="cool"))
        results = store.query({"root_cause": "nonexistent"})
        assert results == []

    def test_query_nonexistent_field(self):
        """Filtering on a field that doesn't exist on the schema returns no results."""
        store = KnowledgeStore(schema=FaultDiagnosis)
        store.insert(FaultDiagnosis(id="F-001", signal_name="temp", root_cause="heat", solution="cool"))
        results = store.query({"made_up_field": "value"})
        assert results == []

    def test_search_full_text(self):
        store = KnowledgeStore(schema=FaultDiagnosis)
        store.insert(FaultDiagnosis(id="F-001", signal_name="engine_temp", root_cause="overheat", solution="replace_fan"))
        store.insert(FaultDiagnosis(id="F-002", signal_name="disk_io", root_cause="fragmentation", solution="defrag"))
        results = store.search("temp")
        assert len(results) == 1
        assert results[0].id == "F-001"

    def test_search_case_insensitive(self):
        store = KnowledgeStore(schema=FaultDiagnosis)
        store.insert(FaultDiagnosis(id="F-001", signal_name="Engine_Temp", root_cause="HEAT", solution="COOL"))
        results = store.search("engine")
        assert len(results) == 1
        assert results[0].id == "F-001"

    def test_search_no_match(self):
        store = KnowledgeStore(schema=FaultDiagnosis)
        store.insert(FaultDiagnosis(id="F-001", signal_name="temp", root_cause="heat", solution="cool"))
        results = store.search("xyzmissing")
        assert results == []

    def test_delete_existing(self):
        store = KnowledgeStore(schema=FaultDiagnosis)
        store.insert(FaultDiagnosis(id="F-001", signal_name="s1", root_cause="r1", solution="sol1"))
        assert store.delete("F-001") is True
        assert store.count() == 0
        assert store.get("F-001") is None

    def test_delete_missing(self):
        store = KnowledgeStore(schema=FaultDiagnosis)
        assert store.delete("nonexistent") is False

    def test_count(self):
        store = KnowledgeStore(schema=FaultDiagnosis)
        assert store.count() == 0
        store.insert(FaultDiagnosis(id="F-001", signal_name="s1", root_cause="r1", solution="sol1"))
        assert store.count() == 1
        store.insert(FaultDiagnosis(id="F-002", signal_name="s2", root_cause="r2", solution="sol2"))
        assert store.count() == 2
        store.delete("F-001")
        assert store.count() == 1

    def test_insert_overwrite(self):
        """Inserting same ID should update existing item."""
        store = KnowledgeStore(schema=FaultDiagnosis)
        fd1 = FaultDiagnosis(id="F-001", signal_name="temp", root_cause="heat", solution="cool")
        store.insert(fd1)
        fd2 = FaultDiagnosis(id="F-001", signal_name="temp", root_cause="heat", solution="replace")
        store.insert(fd2)
        assert store.count() == 1
        assert store.get("F-001").solution == "replace"

    def test_empty_query_returns_all(self):
        store = KnowledgeStore(schema=FaultDiagnosis)
        assert store.query() == []
        store.insert(FaultDiagnosis(id="F-001", signal_name="s1", root_cause="r1", solution="sol1"))
        assert len(store.query()) == 1
