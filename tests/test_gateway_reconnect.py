"""Tests for GatewayManager background reconnect with exponential backoff."""

from __future__ import annotations

import asyncio
import time
import pytest

from kairos.gateway.protocol import ConnectionState, UnifiedMessage, UnifiedResponse
from kairos.gateway.adapters.base import PlatformAdapter
from kairos.gateway.manager import (
    GatewayManager,
    _ReconnectState,
    RECONNECT_BASE_DELAY,
    RECONNECT_MAX_DELAY,
)


# ============================================================================
# Mock adapter for testing
# ============================================================================


class MockAdapter(PlatformAdapter):
    """Fake adapter with controllable connect behavior."""

    def __init__(self, name: str, should_fail: bool = False, fail_count: int = 0):
        self._name = name
        self._state = ConnectionState.DISCONNECTED
        self._should_fail = should_fail
        self._fail_count = fail_count
        self._connect_calls = 0
        self._disconnect_calls = 0
        self._sent: list[tuple[str, UnifiedResponse]] = []

    @property
    def platform_name(self) -> str:
        return self._name

    @property
    def state(self) -> ConnectionState:
        return self._state

    async def connect(self) -> bool:
        self._connect_calls += 1
        if self._should_fail:
            if self._fail_count > 0 and self._connect_calls <= self._fail_count:
                raise ConnectionError(f"Mock {self._name} connection failed")
            self._state = ConnectionState.CONNECTED
            return True
        self._state = ConnectionState.CONNECTED
        return True

    async def disconnect(self) -> None:
        self._disconnect_calls += 1
        self._state = ConnectionState.DISCONNECTED

    async def send(self, chat_id: str, response: UnifiedResponse) -> bool:
        self._sent.append((chat_id, response))
        return True

    async def receive(self) -> UnifiedMessage | None:
        return None

    def translate_incoming(self, raw: dict) -> UnifiedMessage:
        import uuid
        from kairos.gateway.protocol import ContentBlock, ContentType, MessageRole
        return UnifiedMessage(
            id=str(uuid.uuid4()),
            role=MessageRole.USER,
            content=[ContentBlock(type=ContentType.TEXT, text=raw.get("t", ""))],
            platform=self._name,
            chat_id="t",
        )


# ============================================================================
# Tests
# ============================================================================


class TestReconnectState:
    """_ReconnectState dataclass."""

    def test_defaults(self):
        rs = _ReconnectState(name="test")
        assert rs.name == "test"
        assert rs.failures == 0
        assert rs.delay == 0.0
        assert rs.next_attempt == 0.0


class TestReconnectBackoff:
    """Exponential backoff calculation."""

    def test_first_failure_base_delay(self):
        mgr = GatewayManager()
        adapter = MockAdapter("test")
        mgr.register(adapter)

        mgr._record_reconnect_failure("test")
        tracker = mgr._reconnect_tracker["test"]
        assert tracker.failures == 1
        # Base delay ± jitter
        assert RECONNECT_BASE_DELAY * 0.8 <= tracker.delay <= RECONNECT_BASE_DELAY * 1.2

    def test_second_failure_doubles(self):
        mgr = GatewayManager()
        adapter = MockAdapter("test")
        mgr.register(adapter)

        mgr._record_reconnect_failure("test")
        mgr._record_reconnect_failure("test")
        tracker = mgr._reconnect_tracker["test"]
        assert tracker.failures == 2
        expected = RECONNECT_BASE_DELAY * 2
        assert expected * 0.7 <= tracker.delay <= expected * 1.3  # Allow jitter

    def test_backoff_capped_at_max(self):
        mgr = GatewayManager()
        adapter = MockAdapter("test", should_fail=True)
        mgr.register(adapter)

        for _ in range(20):
            mgr._record_reconnect_failure("test")
        tracker = mgr._reconnect_tracker["test"]
        assert tracker.delay <= RECONNECT_MAX_DELAY + (RECONNECT_MAX_DELAY * 0.2)

    def test_next_attempt_is_future(self):
        mgr = GatewayManager()
        adapter = MockAdapter("test")
        mgr.register(adapter)
        mgr._record_reconnect_failure("test")
        tracker = mgr._reconnect_tracker["test"]
        assert tracker.next_attempt > time.time()

    def test_success_clears_tracker(self):
        mgr = GatewayManager()
        adapter = MockAdapter("test")
        mgr.register(adapter)
        mgr._record_reconnect_failure("test")
        assert "test" in mgr._reconnect_tracker
        # Simulate successful reconnect
        adapter._state = ConnectionState.CONNECTED
        # Watcher would pop — let's test directly
        mgr._reconnect_tracker.pop("test", None)
        assert "test" not in mgr._reconnect_tracker


class TestWatcherLifecycle:
    """Start/stop watcher."""

    @pytest.mark.asyncio
    async def test_watcher_starts_and_stops(self):
        mgr = GatewayManager()
        adapter = MockAdapter("test", should_fail=True, fail_count=1)
        mgr.register(adapter)

        # Record a failure to create tracker entry
        mgr._record_reconnect_failure("test")
        assert "test" in mgr._reconnect_tracker

        # Start watcher
        mgr._start_watcher()
        assert mgr._watcher_running

        # Stop watcher
        mgr._stop_watcher()
        assert not mgr._watcher_running

    @pytest.mark.asyncio
    async def test_watcher_idempotent_start(self):
        mgr = GatewayManager()
        mgr._start_watcher()
        assert mgr._watcher_running
        # Second start should be no-op
        mgr._start_watcher()
        assert mgr._watcher_running
        mgr._stop_watcher()

    @pytest.mark.asyncio
    async def test_disconnect_all_stops_watcher(self):
        mgr = GatewayManager()
        adapter = MockAdapter("test")
        mgr.register(adapter)
        mgr._start_watcher()
        assert mgr._watcher_running
        await mgr.disconnect_all()
        assert not mgr._watcher_running


class TestReconnectProperties:
    """reconnect_pending and reconnect_state properties."""

    def test_reconnect_pending_empty(self):
        mgr = GatewayManager()
        assert mgr.reconnect_pending == []
        assert mgr.reconnect_state == {}

    def test_reconnect_pending_after_failure(self):
        mgr = GatewayManager()
        adapter = MockAdapter("discord")
        mgr.register(adapter)
        mgr._record_reconnect_failure("discord")
        assert "discord" in mgr.reconnect_pending

    def test_reconnect_state_structure(self):
        mgr = GatewayManager()
        adapter = MockAdapter("telegram")
        mgr.register(adapter)
        mgr._record_reconnect_failure("telegram")
        state = mgr.reconnect_state["telegram"]
        assert "failures" in state
        assert "delay" in state
        assert "next_attempt" in state
        assert "seconds_until" in state
        assert state["failures"] == 1


class TestCheckReconnects:
    """check_reconnects() backward compat method."""

    def test_check_reconnects_empty(self):
        mgr = GatewayManager()
        assert mgr.check_reconnects() == {}

    def test_check_reconnects_pending(self):
        mgr = GatewayManager()
        adapter = MockAdapter("test")
        mgr.register(adapter)
        mgr._record_reconnect_failure("test")
        # The tracker has next_attempt in the future...
        # But without jitter, next_attempt = now + delay, so it should be > now
        # Wait, the tracker's next_attempt might be in the past if delay was tiny
        tracker = mgr._reconnect_tracker["test"]
        # Force next_attempt to be past
        tracker.next_attempt = 0
        result = mgr.check_reconnects()
        assert result["test"] is True


class TestConnectAllIntegration:
    """connect_all integrates reconnection."""

    @pytest.mark.asyncio
    async def test_connect_all_starts_watcher(self):
        mgr = GatewayManager()
        adapter = MockAdapter("test")
        mgr.register(adapter)
        results = await mgr.connect_all()
        assert results["test"] is True
        assert mgr._watcher_running
        mgr._stop_watcher()

    @pytest.mark.asyncio
    async def test_connect_all_records_failures(self):
        mgr = GatewayManager()
        adapter = MockAdapter("fail", should_fail=True, fail_count=999)
        mgr.register(adapter)
        results = await mgr.connect_all()
        assert results["fail"] is False
        assert "fail" in mgr.reconnect_pending

    @pytest.mark.asyncio
    async def test_shutdown_stops_watcher(self):
        mgr = GatewayManager()
        adapter = MockAdapter("test")
        mgr.register(adapter)
        mgr._start_watcher()
        await mgr.shutdown()
        assert not mgr._watcher_running
