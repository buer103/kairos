"""Tests for CredentialPool — multi-key rotation with rate limit awareness."""

from __future__ import annotations

import time
import threading

from kairos.providers.credential import Credential, CredentialPool, RetryConfig


# ============================================================================
# Credential
# ============================================================================


class TestCredential:
    """Individual credential lifecycle."""

    def test_init_defaults(self):
        c = Credential(key="sk-test123")
        assert c.key == "sk-test123"
        assert c.provider == ""
        assert c.label == ""
        assert c.active is True
        assert c.cooldown_until == 0.0
        assert c.consecutive_failures == 0
        assert c.total_calls == 0
        assert c.rate_limit_hits == 0

    def test_init_with_metadata(self):
        c = Credential(key="sk-abc", provider="openai", label="prod")
        assert c.provider == "openai"
        assert c.label == "prod"

    def test_available_when_fresh(self):
        c = Credential(key="sk-test")
        assert c.available is True

    def test_available_when_disabled(self):
        c = Credential(key="sk-test", active=False)
        assert c.available is False

    def test_available_during_cooldown(self):
        c = Credential(key="sk-test")
        c.cooldown_until = time.time() + 3600  # 1 hour from now
        assert c.available is False

    def test_available_after_cooldown(self):
        c = Credential(key="sk-test")
        c.cooldown_until = time.time() - 1  # 1 second ago
        assert c.available is True

    def test_record_success(self):
        c = Credential(key="sk-test")
        c.consecutive_failures = 2
        c.record_success()
        assert c.consecutive_failures == 0
        assert c.total_calls == 1

    def test_record_failure_below_threshold(self):
        c = Credential(key="sk-test")
        c.record_failure()
        assert c.consecutive_failures == 1
        assert c.active is True

    def test_record_failure_three_strikes_disables(self):
        c = Credential(key="sk-test")
        c.record_failure()
        c.record_failure()
        assert c.active is True
        c.record_failure()
        assert c.active is False
        assert c.consecutive_failures == 3

    def test_record_rate_limit(self):
        c = Credential(key="sk-test")
        c.record_rate_limit(retry_after=5.0)
        assert c.rate_limit_hits == 1
        assert c.consecutive_failures == 1
        assert c.cooldown_until > time.time()
        assert c.cooldown_until <= time.time() + 5.0


# ============================================================================
# CredentialPool
# ============================================================================


class TestCredentialPool:
    """Multi-key credential pool."""

    def test_add_single_key(self):
        pool = CredentialPool()
        cred = pool.add("sk-abc123", provider="openai", label="personal")
        assert cred.key == "sk-abc123"
        assert cred.provider == "openai"
        assert cred.label == "personal"

    def test_add_batch(self):
        pool = CredentialPool()
        creds = pool.add_batch(["sk-1", "sk-2", "sk-3"], provider="openai")
        assert len(creds) == 3
        assert all(c.provider == "openai" for c in creds)

    def test_acquire_single(self):
        pool = CredentialPool()
        pool.add("sk-test", provider="openai")
        cred = pool.acquire("openai")
        assert cred is not None
        assert cred.key == "sk-test"

    def test_acquire_none_for_unknown_provider(self):
        pool = CredentialPool()
        assert pool.acquire("nonexistent") is None

    def test_acquire_returns_best_key(self):
        """acquire prefers key with fewest failures/rate-limit hits."""
        pool = CredentialPool()
        k1 = pool.add("sk-good", provider="openai")
        k2 = pool.add("sk-bad", provider="openai")
        k2.consecutive_failures = 2
        k2.rate_limit_hits = 5
        cred = pool.acquire("openai")
        assert cred is k1

    def test_acquire_skips_cooldown(self):
        pool = CredentialPool()
        k1 = pool.add("sk-cooling", provider="openai")
        k1.cooldown_until = time.time() + 3600
        k2 = pool.add("sk-available", provider="openai")
        cred = pool.acquire("openai")
        assert cred is k2

    def test_acquire_skips_disabled(self):
        pool = CredentialPool()
        k1 = pool.add("sk-disabled", provider="openai")
        k1.active = False
        k2 = pool.add("sk-active", provider="openai")
        cred = pool.acquire("openai")
        assert cred is k2

    def test_acquire_returns_none_when_all_used(self):
        pool = CredentialPool()
        k1 = pool.add("sk-1", provider="openai")
        k1.active = False
        k2 = pool.add("sk-2", provider="openai")
        k2.cooldown_until = time.time() + 3600
        assert pool.acquire("openai") is None

    def test_release_success(self):
        pool = CredentialPool()
        cred = pool.add("sk-test", provider="openai")
        cred.consecutive_failures = 1
        pool.release(cred, success=True)
        assert cred.consecutive_failures == 0
        assert cred.total_calls == 1

    def test_release_failure(self):
        pool = CredentialPool()
        cred = pool.add("sk-test", provider="openai")
        pool.release(cred, success=False)
        assert cred.consecutive_failures == 1

    def test_mark_rate_limited(self):
        pool = CredentialPool()
        cred = pool.add("sk-test", provider="openai")
        pool.mark_rate_limited(cred, retry_after=10.0)
        assert cred.cooldown_until > time.time()
        assert cred.rate_limit_hits == 1

    def test_mark_disabled(self):
        pool = CredentialPool()
        cred = pool.add("sk-test", provider="openai")
        pool.mark_disabled(cred)
        assert cred.active is False

    def test_stats_single_provider(self):
        pool = CredentialPool()
        pool.add("sk-1", provider="openai", label="a")
        pool.add("sk-2", provider="openai", label="b")
        stats = pool.stats(provider="openai")
        assert stats["openai"]["total_keys"] == 2
        assert stats["openai"]["active_keys"] == 2
        assert stats["openai"]["available_keys"] == 2
        assert len(stats["openai"]["keys"]) == 2

    def test_stats_all_providers(self):
        pool = CredentialPool()
        pool.add("sk-1", provider="openai")
        pool.add("sk-2", provider="anthropic")
        stats = pool.stats()
        assert "openai" in stats
        assert "anthropic" in stats
        assert stats["openai"]["total_keys"] == 1
        assert stats["anthropic"]["total_keys"] == 1

    def test_stats_reflects_disabled(self):
        pool = CredentialPool()
        cred = pool.add("sk-1", provider="openai")
        pool.mark_disabled(cred)
        stats = pool.stats(provider="openai")
        assert stats["openai"]["active_keys"] == 0
        assert stats["openai"]["available_keys"] == 0

    def test_stats_reflects_rate_limits(self):
        pool = CredentialPool()
        cred = pool.add("sk-1", provider="openai")
        pool.mark_rate_limited(cred)
        stats = pool.stats(provider="openai")
        assert stats["openai"]["rate_limit_hits"] == 1
        assert stats["openai"]["available_keys"] == 0

    def test_stats_includes_key_details(self):
        pool = CredentialPool()
        pool.add("sk-12345678abcdef", provider="openai", label="my-key")
        stats = pool.stats(provider="openai")
        keys = stats["openai"]["keys"]
        assert len(keys) == 1
        # Has a label, so label is returned directly (not truncated)
        assert keys[0]["label"] == "my-key"
        assert keys[0]["active"] is True
        assert "cooldown_left" in keys[0]

    def test_stats_empty_provider(self):
        pool = CredentialPool()
        stats = pool.stats()
        assert stats == {}

    def test_reset_single_provider(self):
        pool = CredentialPool()
        cred = pool.add("sk-1", provider="openai")
        cred.active = False
        cred.cooldown_until = time.time() + 3600
        cred.consecutive_failures = 3
        pool.reset(provider="openai")
        assert cred.active is True
        assert cred.cooldown_until == 0.0
        assert cred.consecutive_failures == 0

    def test_reset_all_providers(self):
        pool = CredentialPool()
        c1 = pool.add("sk-1", provider="openai")
        c1.active = False
        c2 = pool.add("sk-2", provider="anthropic")
        c2.active = False
        pool.reset()
        assert c1.active is True
        assert c2.active is True

    def test_acquire_release_cycle(self):
        """Full usage cycle: acquire → release(success=True)."""
        pool = CredentialPool()
        pool.add("sk-a", provider="openai")
        pool.add("sk-b", provider="openai")
        c1 = pool.acquire("openai")
        assert c1 is not None
        pool.release(c1, success=True)
        c2 = pool.acquire("openai")
        assert c2 is not None
        pool.release(c2, success=True)

    def test_rate_limit_rotation(self):
        """After marking a key rate-limited, acquire returns another key."""
        pool = CredentialPool()
        pool.add("sk-a", provider="openai")
        pool.add("sk-b", provider="openai")
        c1 = pool.acquire("openai")
        pool.mark_rate_limited(c1, retry_after=60)
        c2 = pool.acquire("openai")
        assert c2 is not c1
        assert c2.key != c1.key

    def test_thread_safe_acquire(self):
        """Multiple threads acquiring should not crash."""
        pool = CredentialPool()
        for i in range(5):
            pool.add(f"sk-{i}", provider="openai")

        errors = []
        def worker():
            try:
                cred = pool.acquire("openai")
                if cred:
                    pool.release(cred, success=True)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


# ============================================================================
# RetryConfig
# ============================================================================


class TestRetryConfig:
    """LLM retry configuration with exponential backoff."""

    def test_defaults(self):
        cfg = RetryConfig()
        assert cfg.max_retries == 3
        assert cfg.base_delay == 1.0
        assert cfg.max_delay == 60.0
        assert cfg.backoff_factor == 2.0
        assert cfg.jitter is True
        assert cfg.retry_on_status == (429, 500, 502, 503, 504)

    def test_custom_config(self):
        cfg = RetryConfig(max_retries=5, base_delay=2.0, max_delay=30.0, jitter=False)
        assert cfg.max_retries == 5
        assert cfg.base_delay == 2.0

    def test_delay_increases_with_attempt(self):
        cfg = RetryConfig(base_delay=1.0, backoff_factor=2.0, jitter=False)
        d0 = cfg.delay_for_attempt(0)   # 1.0 * 2^0 = 1.0
        d1 = cfg.delay_for_attempt(1)   # 1.0 * 2^1 = 2.0
        d2 = cfg.delay_for_attempt(2)   # 1.0 * 2^2 = 4.0
        assert d0 == 1.0
        assert d1 == 2.0
        assert d2 == 4.0

    def test_delay_capped_at_max(self):
        cfg = RetryConfig(base_delay=10.0, backoff_factor=10.0, max_delay=50.0, jitter=False)
        delay = cfg.delay_for_attempt(5)  # 10 * 10^5 = 1,000,000 → capped at 50
        assert delay == 50.0

    def test_delay_with_jitter(self):
        cfg = RetryConfig(base_delay=1.0, jitter=True)
        delays = [cfg.delay_for_attempt(0) for _ in range(20)]
        # With jitter, values should vary
        unique = len(set(round(d, 6) for d in delays))
        # At least some variation
        assert unique > 1, f"Expected jitter variation, got all {delays[0]}"

    def test_delay_without_jitter(self):
        cfg = RetryConfig(base_delay=1.0, jitter=False)
        delays = [cfg.delay_for_attempt(0) for _ in range(5)]
        assert all(d == 1.0 for d in delays)
