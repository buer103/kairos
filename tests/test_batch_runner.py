"""Tests for batch runner."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from kairos.batch_runner import (
    BatchRunner,
    BatchQuery,
    BatchResult,
    BatchSummary,
)


# ═══════════════════════════════════════════════════════════
# BatchResult / BatchSummary
# ═══════════════════════════════════════════════════════════


class TestBatchResult:
    """BatchResult dataclass."""

    def test_success_default(self):
        r = BatchResult(id=0, query="test")
        assert r.success is True
        assert r.error is None

    def test_error_result(self):
        r = BatchResult(id=1, query="test", error="failed")
        assert r.success is False

    def test_total_tokens(self):
        r = BatchResult(id=0, query="test", token_usage={"total_tokens": 100})
        assert r.total_tokens == 100

    def test_default_tokens_zero(self):
        r = BatchResult(id=0, query="test")
        assert r.total_tokens == 0


class TestBatchSummary:
    """BatchSummary aggregation."""

    def test_empty(self):
        s = BatchSummary()
        assert s.success_rate == 1.0
        assert s.avg_duration_ms == 0.0

    def test_mixed_results(self):
        s = BatchSummary(
            total_queries=3,
            completed=2,
            failed=1,
            total_tokens=500,
            total_duration_ms=3000,
            total_tool_calls=5,
        )
        assert s.success_rate == 2 / 3
        assert s.avg_duration_ms == 1500.0
        assert s.avg_tokens == 250.0

    def test_to_dict(self):
        s = BatchSummary(total_queries=2, completed=2, total_duration_ms=1000)
        d = s.to_dict()
        assert d["total_queries"] == 2
        assert d["success_rate"] == 1.0


# ═══════════════════════════════════════════════════════════
# BatchRunner
# ═══════════════════════════════════════════════════════════


class TestBatchRunner:
    """Parallel batch execution."""

    def _mock_agent(self, responses=None):
        """Create a mock agent with configurable responses."""
        agent = MagicMock()
        agent.run.return_value = {
            "content": "mock response",
            "confidence": 0.9,
            "usage": {"total_tokens": 100},
            "evidence": [],
        }
        if responses:
            agent.run.side_effect = responses
        return agent

    def test_empty_queries(self):
        runner = BatchRunner(agent=self._mock_agent())
        summary = runner.run([])
        assert summary.total_queries == 0
        assert summary.completed == 0

    def test_single_query(self):
        agent = self._mock_agent()
        runner = BatchRunner(agent=agent, max_workers=1)
        summary = runner.run(["test query"])

        assert summary.total_queries == 1
        assert summary.completed == 1
        assert summary.failed == 0
        assert len(summary.results) == 1
        assert summary.results[0].content == "mock response"

    def test_multiple_queries_parallel(self):
        agent = self._mock_agent()
        runner = BatchRunner(agent=agent, max_workers=4)
        queries = ["q1", "q2", "q3"]
        summary = runner.run(queries)

        assert summary.total_queries == 3
        assert summary.completed == 3
        assert len(summary.results) == 3
        # Results should be in original order
        assert [r.id for r in summary.results] == [0, 1, 2]
        assert [r.query for r in summary.results] == queries

    def test_error_isolation(self):
        """One failure doesn't stop other queries."""
        agent = self._mock_agent(responses=[
            {"content": "ok1", "usage": {}},
            Exception("failed"),
            {"content": "ok3", "usage": {}},
        ])

        runner = BatchRunner(agent=agent, max_workers=2)
        summary = runner.run(["q1", "q2", "q3"])

        assert summary.total_queries == 3
        assert summary.completed == 2
        assert summary.failed == 1

    def test_progress_callback(self):
        agent = self._mock_agent()
        progress = []

        def cb(completed, total, query):
            progress.append((completed, total))

        runner = BatchRunner(agent=agent, max_workers=2, progress_callback=cb)
        runner.run(["a", "b", "c"])

        assert len(progress) == 3
        assert progress[-1] == (3, 3)

    def test_results_maintain_order(self):
        agent = self._mock_agent(responses=[
            {"content": f"r{i}", "usage": {}} for i in range(3)
        ])
        runner = BatchRunner(agent=agent, max_workers=2)
        summary = runner.run(["q0", "q1", "q2"])

        contents = [r.content for r in summary.results]
        assert contents == ["r0", "r1", "r2"]

    def test_token_aggregation(self):
        agent = self._mock_agent(responses=[
            {"content": "a", "usage": {"total_tokens": 100}},
            {"content": "b", "usage": {"total_tokens": 200}},
        ])
        runner = BatchRunner(agent=agent)
        summary = runner.run(["a", "b"])

        assert summary.total_tokens == 300

    def test_run_with_agent_factory(self):
        """Fresh agent per query."""
        agents_created = []

        def factory():
            agent = MagicMock()
            agent.run.return_value = {"content": "ok", "usage": {}}
            agents_created.append(agent)
            return agent

        runner = BatchRunner(max_workers=2)
        summary = runner.run_with_agent_factory(["a", "b", "c"], factory)

        assert summary.completed == 3
        assert len(agents_created) == 3

    # ── Export ────────────────────────────────────────────────

    def test_export_csv(self):
        s = BatchSummary(
            total_queries=2, completed=2,
            results=[
                BatchResult(id=0, query="q1", content="hello", duration_ms=100),
                BatchResult(id=1, query="q2", content="world", duration_ms=200),
            ],
        )
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            BatchRunner.export_csv(s, f.name)
            content = Path(f.name).read_text()
            Path(f.name).unlink()

        assert "q1" in content
        assert "hello" in content

    def test_export_jsonl(self):
        s = BatchSummary(
            total_queries=1, completed=1,
            results=[
                BatchResult(id=0, query="q1", content="ok", duration_ms=50),
            ],
        )
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            BatchRunner.export_jsonl(s, f.name)
            content = Path(f.name).read_text()
            Path(f.name).unlink()

        assert '"query": "q1"' in content
        assert '"content": "ok"' in content
