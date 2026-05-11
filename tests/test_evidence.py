"""Tests for EvidenceDB — SQLite-backed evidence chain database."""

from __future__ import annotations

import tempfile
from pathlib import Path

from kairos.core.state import Case, Step
from kairos.infra.evidence.tracker import EvidenceDB


# ============================================================================
# EvidenceDB
# ============================================================================


class TestEvidenceDB:
    """Evidence database operations: cases, steps, search, stats."""

    def test_init_default_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = EvidenceDB(db_path=Path(tmpdir) / "test_evidence.db")
            assert db._conn is not None
            db.close()

    def test_tables_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = EvidenceDB(db_path=Path(tmpdir) / "test_evidence.db")
            # Verify tables exist
            tables = db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            table_names = [t[0] for t in tables]
            assert "cases" in table_names
            assert "steps" in table_names
            db.close()

    # ── Save / Load / List / Delete cases ───────────────────────────

    def test_save_case(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = EvidenceDB(db_path=Path(tmpdir) / "test_evidence.db")
            case = Case(id="case-1", conclusion="All good", confidence=0.95)
            db.save(case)
            loaded = db.load("case-1")
            assert loaded is not None
            assert loaded.id == "case-1"
            assert loaded.conclusion == "All good"
            assert loaded.confidence == 0.95
            db.close()

    def test_save_case_with_steps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = EvidenceDB(db_path=Path(tmpdir) / "test_evidence.db")
            case = Case(id="case-2")
            case.add_step("search", {"query": "error logs"})
            case.steps[0].result = {"found": 5}
            db.save(case)
            loaded = db.load("case-2")
            assert loaded is not None
            assert len(loaded.steps) == 1
            assert loaded.steps[0].tool == "search"
            db.close()

    def test_save_case_auto_id(self):
        """save() generates an auto-ID when case has no id attribute."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = EvidenceDB(db_path=Path(tmpdir) / "test_evidence.db")
            # Use a plain object mimicking a case without id
            class SimpleCase:
                id = None  # not set
                confidence = 0.5
                conclusion = "test"
                steps = []

            c = SimpleCase()
            c.id = None
            db.save(c)  # Should not crash
            db.close()

    def test_load_missing_case(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = EvidenceDB(db_path=Path(tmpdir) / "test_evidence.db")
            assert db.load("nonexistent") is None
            db.close()

    def test_list_cases(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = EvidenceDB(db_path=Path(tmpdir) / "test_evidence.db")
            db.save(Case(id="c1", conclusion="ok"))
            db.save(Case(id="c2", conclusion="bad", confidence=0.2))
            cases = db.list_cases(limit=10)
            assert len(cases) == 2
            assert cases[0]["id"] == "c2"  # Most recent first
            assert "steps_count" in cases[0]
            db.close()

    def test_list_cases_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = EvidenceDB(db_path=Path(tmpdir) / "test_evidence.db")
            for i in range(5):
                db.save(Case(id=f"c{i}", conclusion=f"case {i}"))
            cases = db.list_cases(limit=3)
            assert len(cases) == 3
            db.close()

    def test_delete_case(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = EvidenceDB(db_path=Path(tmpdir) / "test_evidence.db")
            db.save(Case(id="to-delete"))
            assert db.load("to-delete") is not None
            db.delete_case("to-delete")
            assert db.load("to-delete") is None
            db.close()

    def test_delete_case_cascades_steps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = EvidenceDB(db_path=Path(tmpdir) / "test_evidence.db")
            case = Case(id="cascade-test")
            case.add_step("tool1", {"arg": 1})
            db.save(case)
            assert len(db.get_steps("cascade-test")) == 1
            db.delete_case("cascade-test")
            assert db.get_steps("cascade-test") == []
            db.close()

    # ── Step operations ─────────────────────────────────────────────

    def test_save_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = EvidenceDB(db_path=Path(tmpdir) / "test_evidence.db")
            db.save(Case(id="case-step"))
            step = Step(id=1, tool="web_search", args={"q": "test"}, duration_ms=150)
            step.case_id = "case-step"
            step.step_id = "step-1"
            step.thread_id = "thread-1"
            step.run_id = "run-1"
            db.save_step(step)
            steps = db.get_steps("case-step")
            assert len(steps) == 1
            s = steps[0]
            assert s["step_id"] == "step-1"
            assert s["tool"] == "web_search"
            assert s["thread_id"] == "thread-1"
            assert s["run_id"] == "run-1"
            db.close()

    def test_save_step_with_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = EvidenceDB(db_path=Path(tmpdir) / "test_evidence.db")
            db.save(Case(id="case-err"))
            step = Step(id=1, tool="failing_tool", args={})
            step.case_id = "case-err"
            step.step_id = "err-1"
            step.error = "Connection timeout"
            db.save_step(step)
            steps = db.get_steps("case-err")
            assert len(steps) == 1
            assert steps[0]["error"] == "Connection timeout"
            db.close()

    def test_save_step_auto_step_id(self):
        """save_step generates step_id when not provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = EvidenceDB(db_path=Path(tmpdir) / "test_evidence.db")
            db.save(Case(id="case-auto"))
            step = Step(id=1, tool="test", args={})
            step.case_id = "case-auto"
            db.save_step(step)  # No step_id set
            steps = db.get_steps("case-auto")
            assert len(steps) == 1
            assert steps[0]["step_id"] is not None
            db.close()

    def test_get_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = EvidenceDB(db_path=Path(tmpdir) / "test_evidence.db")
            db.save(Case(id="case-gs"))
            step = Step(id=1, tool="grep", args={"pattern": "error"})
            step.case_id = "case-gs"
            step.step_id = "specific-1"
            db.save_step(step)
            found = db.get_step("case-gs", "specific-1")
            assert found is not None
            assert found["tool"] == "grep"
            db.close()

    def test_get_step_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = EvidenceDB(db_path=Path(tmpdir) / "test_evidence.db")
            assert db.get_step("no-case", "no-step") is None
            db.close()

    def test_get_steps_ordered(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = EvidenceDB(db_path=Path(tmpdir) / "test_evidence.db")
            db.save(Case(id="case-order"))
            for i in range(3):
                step = Step(id=i, tool=f"tool_{i}", args={})
                step.case_id = "case-order"
                step.step_id = f"step-{i}"
                db.save_step(step)
            steps = db.get_steps("case-order")
            assert len(steps) == 3
            # Should be ordered by timestamp ascending
            ts_list = [s["timestamp"] for s in steps]
            assert ts_list == sorted(ts_list)
            db.close()

    # ── Search steps ────────────────────────────────────────────────

    def test_search_by_tool(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = EvidenceDB(db_path=Path(tmpdir) / "test_evidence.db")
            db.save(Case(id="case-s1"))
            s1 = Step(id=1, tool="web_search", args={})
            s1.case_id = "case-s1"; s1.step_id = "s1"
            s2 = Step(id=2, tool="web_scrape", args={})
            s2.case_id = "case-s1"; s2.step_id = "s2"
            db.save_step(s1)
            db.save_step(s2)
            results = db.search_steps(tool="web_search")
            assert len(results) == 1
            assert results[0]["tool"] == "web_search"
            db.close()

    def test_search_by_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = EvidenceDB(db_path=Path(tmpdir) / "test_evidence.db")
            db.save(Case(id="case-err-s"))
            ok = Step(id=1, tool="ok_tool", args={})
            ok.case_id = "case-err-s"; ok.step_id = "ok"
            err = Step(id=2, tool="bad_tool", args={})
            err.case_id = "case-err-s"; err.step_id = "err"; err.error = "timeout"
            db.save_step(ok)
            db.save_step(err)
            results = db.search_steps(error=True)
            assert len(results) == 1
            assert results[0]["error"] == "timeout"
            db.close()

    def test_search_by_thread_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = EvidenceDB(db_path=Path(tmpdir) / "test_evidence.db")
            db.save(Case(id="case-th"))
            s1 = Step(id=1, tool="t1", args={})
            s1.case_id = "case-th"; s1.step_id = "s1"; s1.thread_id = "thread-A"
            s2 = Step(id=2, tool="t2", args={})
            s2.case_id = "case-th"; s2.step_id = "s2"; s2.thread_id = "thread-B"
            db.save_step(s1)
            db.save_step(s2)
            results = db.search_steps(thread_id="thread-A")
            assert len(results) == 1
            assert results[0]["thread_id"] == "thread-A"
            db.close()

    def test_search_since_timestamp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = EvidenceDB(db_path=Path(tmpdir) / "test_evidence.db")
            db.save(Case(id="case-time"))
            s1 = Step(id=1, tool="old", args={})
            s1.case_id = "case-time"; s1.step_id = "old-step"
            db.save_step(s1)
            import time
            mid = time.time()
            s2 = Step(id=2, tool="new", args={})
            s2.case_id = "case-time"; s2.step_id = "new-step"
            db.save_step(s2)
            results = db.search_steps(since=mid)
            assert len(results) >= 1
            assert results[0]["tool"] == "new"
            db.close()

    def test_search_combined_filters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = EvidenceDB(db_path=Path(tmpdir) / "test_evidence.db")
            db.save(Case(id="case-combo"))
            s1 = Step(id=1, tool="search", args={})
            s1.case_id = "case-combo"; s1.step_id = "s1"; s1.error = "fail"
            s2 = Step(id=2, tool="other", args={})
            s2.case_id = "case-combo"; s2.step_id = "s2"; s2.error = "fail"
            db.save_step(s1)
            db.save_step(s2)
            results = db.search_steps(tool="search", error=True)
            assert len(results) == 1
            assert results[0]["tool"] == "search"
            assert results[0]["error"] == "fail"
            db.close()

    # ── Stats ───────────────────────────────────────────────────────

    def test_stats_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = EvidenceDB(db_path=Path(tmpdir) / "test_evidence.db")
            stats = db.stats()
            assert stats["cases"] == 0
            assert stats["steps"] == 0
            assert stats["errors"] == 0
            assert stats["error_rate"] == 0
            db.close()

    def test_stats_with_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = EvidenceDB(db_path=Path(tmpdir) / "test_evidence.db")
            db.save(Case(id="c1"))
            db.save(Case(id="c2"))
            s1 = Step(id=1, tool="ok", args={})
            s1.case_id = "c1"; s1.step_id = "s1"
            s2 = Step(id=2, tool="bad", args={})
            s2.case_id = "c1"; s2.step_id = "s2"; s2.error = "fail"
            db.save_step(s1)
            db.save_step(s2)
            stats = db.stats()
            assert stats["cases"] == 2
            assert stats["steps"] == 2
            assert stats["errors"] == 1
            assert stats["error_rate"] == 50.0
            db.close()

    # ── Close ───────────────────────────────────────────────────────

    def test_close(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = EvidenceDB(db_path=Path(tmpdir) / "test_evidence.db")
            db.close()
            # Closing again should raise (sqlite3.ProgrammingError)
            import sqlite3
            try:
                db._conn.execute("SELECT 1")
                assert False, "Should have raised"
            except sqlite3.ProgrammingError:
                pass
