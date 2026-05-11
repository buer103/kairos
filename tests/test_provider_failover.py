"""Tests for provider failover — multi-backend degradation and auto-recovery."""

from __future__ import annotations

import time
import threading

import pytest

from kairos.core.loop import ModelHealth
from kairos.providers.base import ModelConfig
from kairos.providers.failover import (
    ProviderFailoverManager,
    ProviderHealth,
    build_failover_from_env,
)


# ═══════════════════════════════════════════════════════════
# ModelHealth
# ═══════════════════════════════════════════════════════════


class TestModelHealth:
    """Health tracking for individual providers."""

    def test_initial_healthy(self):
        h = ModelHealth()
        assert h.is_healthy is True
        assert h.consecutive_failures == 0
        assert h.failure_rate == 0.0

    def test_record_success_resets(self):
        h = ModelHealth()
        h.consecutive_failures = 2
        h.record_success()
        assert h.consecutive_failures == 0
        assert h.is_healthy is True

    def test_three_failures_unhealthy(self):
        h = ModelHealth()
        h.record_failure(ErrorKind.UNKNOWN)
        h.record_failure(ErrorKind.UNKNOWN)
        assert h.is_healthy is True
        h.record_failure(ErrorKind.UNKNOWN)
        assert h.is_healthy is False

    def test_rate_limit_sets_cooldown(self):
        h = ModelHealth()
        t0 = time.time()
        h.record_failure(ErrorKind.RATE_LIMIT)
        assert h.cooldown_until > t0
        assert h.cooldown_until <= t0 + 35  # ~30s default

    def test_cooldown_block(self):
        h = ModelHealth()
        h.cooldown_until = time.time() + 3600
        assert h.is_healthy is False

    def test_cooldown_expired(self):
        h = ModelHealth()
        h.cooldown_until = time.time() - 1
        # Even with past cooldown, 0 consecutive failures = healthy
        assert h.is_healthy is True

    def test_failure_rate(self):
        h = ModelHealth()
        h.total_calls = 10
        h.total_failures = 3
        assert h.failure_rate == 0.3

    def test_failure_rate_zero_calls(self):
        h = ModelHealth()
        assert h.failure_rate == 0.0


# ═══════════════════════════════════════════════════════════
# ProviderHealth (standalone)
# ═══════════════════════════════════════════════════════════


class TestProviderHealth:
    """Standalone provider health tracking."""

    def test_initial_available(self):
        ph = ProviderHealth(provider_key="test")
        assert ph.available is True
        assert ph.disabled is False

    def test_record_success(self):
        ph = ProviderHealth(provider_key="test")
        ph.consecutive_failures = 2
        ph.record_success()
        assert ph.consecutive_failures == 0
        assert ph.total_calls == 1

    def test_record_failure(self):
        ph = ProviderHealth(provider_key="test")
        ph.record_failure()
        assert ph.consecutive_failures == 1
        assert ph.total_calls == 1

    def test_disabled_not_available(self):
        ph = ProviderHealth(provider_key="test", disabled=True)
        assert ph.available is False

    def test_cooldown_not_available(self):
        ph = ProviderHealth(provider_key="test")
        ph.cooldown_until = time.time() + 3600
        assert ph.available is False

    def test_record_failover_use(self):
        ph = ProviderHealth(provider_key="test")
        ph.record_failover_use()
        assert ph.failover_count == 1
        assert ph.total_calls == 1


# ═══════════════════════════════════════════════════════════
# ProviderFailoverManager
# ═══════════════════════════════════════════════════════════


class TestProviderFailoverManager:
    """Multi-backend failover with cooldown and auto-recovery."""

    def _make_configs(self):
        primary = ModelConfig(api_key="sk-" + "p" * 20, model="primary-model", base_url="https://primary.api/v1")
        fb1 = ModelConfig(api_key="sk-" + "f" * 20, model="fallback-1", base_url="https://fb1.api/v1")
        fb2 = ModelConfig(api_key="sk-" + "g" * 20, model="fallback-2", base_url="https://fb2.api/v1")
        return primary, fb1, fb2

    def test_init_with_primary_only(self):
        primary, _, _ = self._make_configs()
        mgr = ProviderFailoverManager(primary=primary)
        assert mgr.current == primary
        assert mgr.is_failover_active is False
        assert mgr.failover_count == 0

    def test_init_with_fallbacks(self):
        primary, fb1, fb2 = self._make_configs()
        mgr = ProviderFailoverManager(primary=primary, fallbacks=[fb1, fb2])
        assert mgr.current == primary

    def test_mark_failure_triggers_failover(self):
        primary, fb1, _ = self._make_configs()
        mgr = ProviderFailoverManager(
            primary=primary, fallbacks=[fb1],
            max_consecutive_failures=2,
        )
        # 1 failure: still on primary
        result = mgr.mark_failure()
        assert result is False
        assert mgr.current == primary

        # 2 failures: failover to fb1
        result = mgr.mark_failure()
        assert result is True
        assert mgr.is_failover_active is True
        assert mgr.current == fb1

    def test_mark_success_on_fallback(self):
        primary, fb1, _ = self._make_configs()
        mgr = ProviderFailoverManager(
            primary=primary, fallbacks=[fb1],
            max_consecutive_failures=1,
        )
        mgr.mark_failure()  # triggers failover
        assert mgr.current == fb1
        mgr.mark_success()
        assert mgr.is_failover_active is True  # still on fallback

    def test_auto_recover_to_primary(self):
        """Primary recovers after cooldown."""
        primary, fb1, _ = self._make_configs()
        mgr = ProviderFailoverManager(
            primary=primary, fallbacks=[fb1],
            cooldown_seconds=0.1,  # 100ms cooldown
            max_consecutive_failures=1,
        )
        mgr.mark_failure()  # primary fails, switch to fb1
        assert mgr.current == fb1

        time.sleep(0.15)  # Wait for cooldown
        config = mgr.acquire()  # acquire tries recovery
        assert config == primary
        assert mgr.is_failover_active is False

    def test_fallback_to_second(self):
        primary, fb1, fb2 = self._make_configs()
        mgr = ProviderFailoverManager(
            primary=primary, fallbacks=[fb1, fb2],
            max_consecutive_failures=1,
        )
        mgr.mark_failure()  # primary → fb1
        assert mgr.current == fb1
        mgr.mark_failure()  # fb1 → fb2 (primary still in cooldown)
        assert mgr.current == fb2

    def test_mark_rate_limited(self):
        primary, fb1, _ = self._make_configs()
        mgr = ProviderFailoverManager(primary=primary, fallbacks=[fb1])
        result = mgr.mark_rate_limited(retry_after=60)
        assert result is True
        assert mgr.current == fb1

    def test_reset(self):
        primary, fb1, _ = self._make_configs()
        mgr = ProviderFailoverManager(
            primary=primary, fallbacks=[fb1],
            max_consecutive_failures=1,
        )
        mgr.mark_failure()
        assert mgr.current == fb1

        mgr.reset()
        assert mgr.current == primary
        assert mgr.is_failover_active is False

    def test_stats(self):
        primary, fb1, _ = self._make_configs()
        mgr = ProviderFailoverManager(
            primary=primary, fallbacks=[fb1],
            max_consecutive_failures=1,
        )
        mgr.mark_failure()
        mgr.mark_failure()  # triggers failover

        stats = mgr.stats()
        assert stats["is_failover"] is True
        assert stats["chain_length"] == 2
        assert "providers" in stats

    def test_all_down_stays_on_last(self):
        """When all providers fail, stays on the last one."""
        primary, fb1, _ = self._make_configs()
        mgr = ProviderFailoverManager(
            primary=primary, fallbacks=[fb1],
            max_consecutive_failures=1,
            cooldown_seconds=3600,
        )
        mgr.mark_failure()  # primary → fb1
        mgr.mark_failure()  # fb1 fails, nothing left
        assert mgr.current == fb1  # stays on last

    def test_acquire_returns_current(self):
        primary, _, _ = self._make_configs()
        mgr = ProviderFailoverManager(primary=primary)
        assert mgr.acquire() == primary

    def test_thread_safety(self):
        """Concurrent access doesn't corrupt state."""
        primary, fb1, fb2 = self._make_configs()
        mgr = ProviderFailoverManager(
            primary=primary, fallbacks=[fb1, fb2],
            max_consecutive_failures=5,
        )

        errors = []
        def worker():
            try:
                for _ in range(20):
                    cfg = mgr.acquire()
                    mgr.mark_success()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


# ═══════════════════════════════════════════════════════════
# Helper imports
# ═══════════════════════════════════════════════════════════

from kairos.observability.error_classifier import ErrorKind
