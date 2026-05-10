"""Tests for kairos.memory.tiers — three-tier memory with confidence filtering."""

import time

import pytest

from kairos.memory.tiers import (
    CONFIDENCE_THRESHOLD,
    MAX_INJECTION_TOKENS,
    ConfidenceFilter,
    MemoryEntry,
    MemoryTier,
    TieredMemoryStore,
)


class TestConfidenceFilter:
    """Tests for ConfidenceFilter threshold logic."""

    def test_default_threshold(self):
        cf = ConfidenceFilter()
        assert cf.threshold == CONFIDENCE_THRESHOLD
        assert cf.passes(0.8) is True
        assert cf.passes(0.7) is True
        assert cf.passes(0.69) is False

    def test_custom_threshold(self):
        cf = ConfidenceFilter(threshold=0.9)
        assert cf.passes(0.9) is True
        assert cf.passes(0.89) is False

    def test_filter_entries(self):
        cf = ConfidenceFilter(threshold=0.7)
        entries = [
            MemoryEntry(key="a", value="", tier=MemoryTier.FACT, confidence=0.9),
            MemoryEntry(key="b", value="", tier=MemoryTier.FACT, confidence=0.5),
            MemoryEntry(key="c", value="", tier=MemoryTier.FACT, confidence=0.7),
        ]
        filtered = cf.filter(entries)
        assert len(filtered) == 2
        assert {e.key for e in filtered} == {"a", "c"}


class TestMemoryEntry:
    """Tests for MemoryEntry dataclass."""

    def test_to_dict(self):
        entry = MemoryEntry(
            key="k", value="v", tier=MemoryTier.FACT,
            agent_id="test", category="fact", confidence=0.85,
        )
        d = entry.to_dict()
        assert d["key"] == "k"
        assert d["tier"] == "fact"
        assert d["agent_id"] == "test"
        assert d["confidence"] == 0.85

    def test_is_expired(self):
        entry = MemoryEntry(
            key="k", value="v", tier=MemoryTier.FACT,
            created_at=1000.0, ttl=100.0,
        )
        assert entry.is_expired(now=1100.0) is False  # 100s elapsed = exactly at TTL
        assert entry.is_expired(now=1101.0) is True   # over TTL

    def test_is_expired_no_ttl(self):
        entry = MemoryEntry(key="k", value="v", tier=MemoryTier.FACT, ttl=None)
        assert entry.is_expired(now=999999.0) is False

    def test_meets_confidence(self):
        entry = MemoryEntry(key="k", value="v", tier=MemoryTier.FACT, confidence=0.7)
        assert entry.meets_confidence(0.7) is True
        assert entry.meets_confidence(0.71) is False


class TestTieredMemoryStore:
    """Tests for the three-tier memory store."""

    @pytest.fixture
    def store(self):
        """Fresh in-memory store for each test."""
        from kairos.memory.backends import DictBackend
        return TieredMemoryStore(backend=DictBackend())

    # ---- Profile tier ----

    def test_save_and_get_profile(self, store):
        store.save_profile("lang", "zh-CN")
        entry = store.get_profile("lang")
        assert entry is not None
        assert entry.value == "zh-CN"
        assert entry.tier == MemoryTier.PROFILE

    def test_profile_overwrites(self, store):
        store.save_profile("lang", "zh-CN")
        store.save_profile("lang", "en-US")
        entry = store.get_profile("lang")
        assert entry.value == "en-US"

    def test_list_profiles(self, store):
        store.save_profile("lang", "zh-CN")
        store.save_profile("timezone", "Asia/Shanghai")
        profiles = store.list_profiles()
        assert len(profiles) == 2

    # ---- Timeline tier ----

    def test_append_timeline(self, store):
        store.append_timeline("code_review", "Reviewed PR #1")
        store.append_timeline("merged", "Merged PR #2")
        timeline = store.get_timeline()
        assert len(timeline) == 2
        # Most recent first
        assert "PR #2" in timeline[0].value or "Merged" in timeline[0].value

    def test_timeline_event_type_filter(self, store):
        store.append_timeline("code_review", "PR #1")
        store.append_timeline("deploy", "Deployed v0.15")
        code = store.get_timeline(event_type="code_review")
        deploy = store.get_timeline(event_type="deploy")
        assert len(code) == 1
        assert len(deploy) == 1

    def test_timeline_limit(self, store):
        for i in range(10):
            store.append_timeline(f"event_{i}", f"Event {i}")
        results = store.get_timeline(limit=3)
        assert len(results) == 3

    # ---- Fact tier ----

    def test_save_fact_above_threshold(self, store):
        ok = store.save_fact("project", "Kairos v0.15", confidence=0.9)
        assert ok is True
        entry = store.get_fact("project")
        assert entry is not None
        assert entry.value == "Kairos v0.15"

    def test_save_fact_below_threshold_rejected(self, store):
        ok = store.save_fact("uncertain", "maybe true", confidence=0.3)
        assert ok is False
        entry = store.get_fact("uncertain")
        assert entry is None

    def test_save_fact_at_threshold(self, store):
        ok = store.save_fact("edge", "at 0.7", confidence=0.7)
        assert ok is True

    def test_save_fact_invalid_category_warns(self, store):
        ok = store.save_fact("test", "value", confidence=0.8, category="invalid_cat")
        assert ok is True
        entry = store.get_fact("test")
        assert entry.category == "fact"  # fell back to 'fact'

    def test_list_facts_by_category(self, store):
        store.save_fact("a", "pref val", confidence=0.9, category="preference")
        store.save_fact("b", "fact val", confidence=0.8, category="fact")
        store.save_fact("c", "knowledge val", confidence=0.75, category="knowledge")

        prefs = store.list_facts(category="preference")
        facts = store.list_facts(category="fact")
        assert len(prefs) == 1
        assert len(facts) == 1

    def test_list_facts_min_confidence(self, store):
        store.save_fact("a", "high", confidence=0.95)
        store.save_fact("b", "medium", confidence=0.75)
        store.save_fact("c", "below", confidence=0.65)
        store.save_fact("d", "very low", confidence=0.5)

        high = store.list_facts(min_confidence=0.9)
        assert len(high) == 1
        all_above = store.list_facts(min_confidence=0.7)
        assert len(all_above) == 2

    # ---- Per-agent isolation ----

    def test_per_agent_isolation(self, store):
        store.save_profile("lang", "zh-CN", agent_id="alice")
        store.save_profile("lang", "en-US", agent_id="bob")

        assert store.get_profile("lang", "alice").value == "zh-CN"
        assert store.get_profile("lang", "bob").value == "en-US"

    def test_clear_agent(self, store):
        store.save_profile("lang", "zh-CN", agent_id="alice")
        store.save_fact("project", "Kairos", confidence=0.9, agent_id="alice")
        store.save_profile("lang", "en", agent_id="bob")

        deleted = store.clear_agent("alice")
        assert deleted == 2
        assert store.get_profile("lang", "alice") is None
        assert store.get_profile("lang", "bob") is not None  # bob untouched

    # ---- Search ----

    def test_search_across_tiers(self, store):
        store.save_profile("lang", "zh-CN")
        store.save_fact("version", "Kairos v0.15", confidence=0.9)
        store.append_timeline("launch", "Kairos launched")

        results = store.search("Kairos")
        assert len(results) >= 1
        values = {r.value for r in results}
        assert "Kairos v0.15" in values

    def test_search_respects_confidence(self, store):
        store.save_fact("project", "Kairos", confidence=0.9)
        store.save_fact("uncertain", "maybe not Kairos", confidence=0.3)

        results = store.search("Kairos")
        values = {r.value for r in results}
        assert "Kairos" in values
        assert "maybe not Kairos" not in values  # rejected

    def test_search_respects_agent_id(self, store):
        store.save_fact("project", "Kairos Alice", confidence=0.9, agent_id="alice")
        store.save_fact("project", "Kairos Bob", confidence=0.9, agent_id="bob")

        results = store.search("Kairos", agent_id="alice")
        values = {r.value for r in results}
        assert "Kairos Alice" in values
        assert "Kairos Bob" not in values

    # ---- Prompt formatting ----

    def test_format_for_prompt(self, store):
        store.save_profile("lang", "zh-CN")
        store.save_fact("version", "Kairos v0.15", confidence=0.9)
        store.append_timeline("code_review", "Reviewed PR #42")

        block = store.format_for_prompt()
        assert "## USER PROFILE" in block
        assert "zh-CN" in block
        assert "## FACTS" in block
        assert "Kairos v0.15" in block
        assert "## TIMELINE" in block

    def test_format_for_prompt_empty(self, store):
        block = store.format_for_prompt()
        assert block == ""

    def test_format_for_prompt_token_budget(self, store):
        store.save_profile("lang", "zh-CN")
        block = store.format_for_prompt(max_tokens=1)  # Very small budget
        # Should still work without crashing
        assert isinstance(block, str)

    def test_format_for_prompt_selective_tiers(self, store):
        store.save_profile("lang", "zh-CN")
        store.save_fact("version", "Kairos", confidence=0.9)
        block = store.format_for_prompt(include_timeline=False, include_facts=False)
        assert "## USER PROFILE" in block
        assert "## FACTS" not in block
        assert "## TIMELINE" not in block

    # ---- Stats ----

    def test_stats(self, store):
        store.save_profile("lang", "zh-CN")
        store.save_fact("a", "high", confidence=0.9)
        store.save_fact("b", "low", confidence=0.5)  # rejected
        store.append_timeline("event", "something")

        stats = store.stats()
        assert stats["profile_count"] == 1
        assert stats["timeline_count"] >= 1
        assert stats["fact_count"] == 1  # only the high-conf one
        assert stats["fact_above_threshold"] == 1


class TestMemoryTierEnum:
    """Enum value tests."""

    def test_tier_values(self):
        assert MemoryTier.PROFILE.value == "profile"
        assert MemoryTier.TIMELINE.value == "timeline"
        assert MemoryTier.FACT.value == "fact"

    def test_tier_from_string(self):
        assert MemoryTier("profile") == MemoryTier.PROFILE
        assert MemoryTier("timeline") == MemoryTier.TIMELINE
        assert MemoryTier("fact") == MemoryTier.FACT

    def test_tier_invalid_raises(self):
        with pytest.raises(ValueError):
            MemoryTier("invalid")


class TestConstants:
    """Verify default constants are sensible."""

    def test_defaults(self):
        assert CONFIDENCE_THRESHOLD == 0.7
        assert MAX_INJECTION_TOKENS == 2000

    def test_fact_categories(self):
        from kairos.memory.tiers import FACT_CATEGORIES
        assert "preference" in FACT_CATEGORIES
        assert "fact" in FACT_CATEGORIES
        assert "knowledge" in FACT_CATEGORIES
        assert "decision" in FACT_CATEGORIES
        assert "action" in FACT_CATEGORIES
