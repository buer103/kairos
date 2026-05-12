"""Tests for pluaggable Context Engine protocol."""
from __future__ import annotations

import pytest

from kairos.engines import ContextEngine
from kairos.core.state import ThreadState


class _MinimalEngine(ContextEngine):
    """Minimal engine that implements compress."""

    def compress(self, state, runtime, budget_ratio=0.8):
        return state.messages, {"engine": self.name, "before": len(state.messages), "after": len(state.messages)}


class _TruncatingEngine(ContextEngine):
    """Engine that keeps only last N messages."""

    def compress(self, state, runtime, budget_ratio=0.8):
        keep = max(1, int(len(state.messages) * budget_ratio))
        compressed = state.messages[-keep:]
        return compressed, {"engine": self.name, "before": len(state.messages), "after": len(compressed)}

    @property
    def name(self):
        return "TruncatingEngine"


def test_minimal_engine_implements_protocol():
    engine = _MinimalEngine()
    state = ThreadState()
    state.messages = [{"role": "user", "content": "hello"}]

    msgs, stats = engine.compress(state, {}, 0.8)
    assert msgs == state.messages
    assert stats["engine"] == "_MinimalEngine"
    assert stats["before"] == 1
    assert stats["after"] == 1


def test_truncating_engine():
    engine = _TruncatingEngine()
    state = ThreadState()
    state.messages = [
        {"role": "user", "content": f"msg{i}"} for i in range(10)
    ]

    msgs, stats = engine.compress(state, {}, 0.5)
    assert len(msgs) == 5
    assert stats["engine"] == "TruncatingEngine"
    assert stats["before"] == 10
    assert stats["after"] == 5


def test_engine_has_name():
    engine = _MinimalEngine()
    assert engine.name == "_MinimalEngine"


def test_hooks_called():
    """on_compress_start and on_compress_end are called."""
    called = []

    class HookedEngine(ContextEngine):
        def compress(self, state, runtime, budget_ratio=0.8):
            return state.messages, {"engine": self.name, "before": 0, "after": 0}

        def on_compress_start(self, state, runtime):
            called.append("start")

        def on_compress_end(self, state, runtime, stats):
            called.append("end")

    engine = HookedEngine()
    state = ThreadState()
    state.messages = []

    engine.on_compress_start(state, {})
    engine.compress(state, {}, 0.8)
    engine.on_compress_end(state, {}, {})

    assert called == ["start", "end"]


def test_engine_is_abstract():
    """Cannot instantiate ContextEngine directly."""
    with pytest.raises(TypeError):
        ContextEngine()  # type: ignore


def test_engine_import_from_package():
    """ContextEngine is importable from kairos.engines."""
    from kairos.engines import ContextEngine as CE
    assert CE is ContextEngine
