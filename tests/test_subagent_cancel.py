"""Tests for CancelEvent and sub-agent cancellation."""

import threading
import time
import pytest
from unittest.mock import MagicMock

from kairos.agents.delegate import (
    CancelEvent, SubAgent, DelegateTask, DelegationManager, DelegateConfig
)


class TestCancelEvent:
    """Unit tests for CancelEvent."""

    def test_default_state(self):
        ce = CancelEvent()
        assert ce.is_set() is False
        assert ce.cancelled is False
        assert ce.reason == "Cancelled"

    def test_set_and_check(self):
        ce = CancelEvent(reason="test")
        ce.set()
        assert ce.is_set() is True
        assert ce.cancelled is True
        assert ce.set_at > 0

    def test_set_with_reason(self):
        ce = CancelEvent()
        ce.set(reason="timeout")
        assert ce.reason == "timeout"

    def test_clear(self):
        ce = CancelEvent()
        ce.set(reason="test")
        ce.clear()
        assert ce.is_set() is False
        assert ce.set_at == 0

    def test_wait_timeout(self):
        ce = CancelEvent()
        result = ce.wait(timeout=0.01)
        assert result is False  # not set within timeout

    def test_wait_signaled(self):
        ce = CancelEvent()

        def signal():
            time.sleep(0.02)
            ce.set()

        t = threading.Thread(target=signal)
        t.start()
        result = ce.wait(timeout=0.2)
        t.join()
        assert result is True

    def test_repr(self):
        ce = CancelEvent()
        assert "CancelEvent" in repr(ce)
        assert "clear" in repr(ce)

        ce.set(reason="done")
        assert "set" in repr(ce)
        assert "done" in repr(ce)


class TestSubAgentCancellation:
    """Tests for sub-agent with cancel_event."""

    def test_subagent_cancelled_before_start(self):
        """When cancel_event is set before run(), returns cancellation result."""
        task = DelegateTask(goal="test")
        model = MagicMock()
        cancel = CancelEvent()
        cancel.set(reason="test cancel")

        sub = SubAgent(task=task, model_provider=model, cancel_event=cancel)
        result = sub.run()

        assert result.success is False
        assert "Cancelled" in result.error
        assert result.task_id == task.id

    def test_subagent_runs_normally_when_not_cancelled(self):
        """When cancel_event is not set, sub-agent runs normally."""
        # This requires a real model call, so skip in unit test
        pass


class TestDelegationManagerCancellation:
    """Tests for DelegationManager cancel_all()."""

    def test_cancel_all_no_active_agents(self):
        """cancel_all with no active agents returns 0."""
        mgr = DelegationManager(model=MagicMock())
        count = mgr.cancel_all("test")
        assert count == 0
        assert mgr.cancelled is True

    def test_cancel_all_sets_global_flag(self):
        """cancel_all sets the global cancel flag."""
        mgr = DelegationManager(model=MagicMock())
        mgr.cancel_all("shutdown")
        assert mgr.cancelled is True

    def test_active_count_default_zero(self):
        """Default active_count is 0."""
        mgr = DelegationManager(model=MagicMock())
        assert mgr.active_count == 0

    def test_cancel_event_on_timeout(self):
        """_run_with_timeout sets cancel_event on timeout."""
        task = DelegateTask(goal="slow", timeout=0.01)
        model = MagicMock()

        # Create a sub-agent that will hang
        cancel = CancelEvent()
        sub = SubAgent(task=task, model_provider=model, cancel_event=cancel)

        def slow_run():
            # Simulate a very slow operation
            time.sleep(10)
            return MagicMock(content="done")

        sub.run = slow_run  # override to hang

        mgr = DelegationManager(model=model, config=DelegateConfig(default_timeout=0.02))
        result = mgr._run_with_timeout(sub, timeout=0.02)

        assert result.success is False
        assert "Timed out" in result.error
        assert cancel.is_set() is True  # cancel event was signaled
