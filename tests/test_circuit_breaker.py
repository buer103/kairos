"""Tests for 3-state circuit breaker + error classifier (DeerFlow-compatible).

Covers: CircuitBreaker (3-state), ErrorClassifier (5 categories + CN patterns).
"""

from __future__ import annotations

import time
import threading
import pytest

from kairos.middleware.llm_retry import (
    CircuitBreaker,
    CircuitState,
    ErrorCategory,
    ErrorClassifier,
    ClassifiedError,
)


# ============================================================================
# CircuitBreaker — 3-state
# ============================================================================


class TestCircuitBreakerStates:
    """Verify 3-state lifecycle: CLOSED → OPEN → HALF_OPEN → CLOSED."""

    def test_initial_state_closed(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert not cb.is_open

    def test_closed_accepts_calls(self):
        cb = CircuitBreaker()
        assert cb.before_call() is True

    def test_trip_to_open(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.is_open

    def test_open_rejects_calls(self):
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.before_call() is False

    def test_half_open_after_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.05)
        # is_open should return False and transition to HALF_OPEN
        assert not cb.is_open
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_probe_success(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.01)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.05)
        assert not cb.is_open
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_probe_failure_back_to_open(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.01)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.05)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_reset_from_open(self):
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb._failures == 0

    def test_reset_from_half_open(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        time.sleep(0.05)
        assert cb.state == CircuitState.HALF_OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED

    def test_partial_failures_no_trip(self):
        cb = CircuitBreaker(failure_threshold=5)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        assert cb.before_call() is True

    def test_success_resets_counter(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        # Counter reset
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED  # Still only 2 failures since reset


class TestCircuitBreakerThreadSafety:
    """Verify thread-safe state transitions."""

    def test_concurrent_failures(self):
        cb = CircuitBreaker(failure_threshold=10)
        errors = []

        def fail():
            try:
                for _ in range(5):
                    cb.record_failure()
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=fail)
        t2 = threading.Thread(target=fail)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, f"Thread errors: {errors}"
        # 10 failures should trip
        assert cb.state == CircuitState.OPEN

    def test_concurrent_reset_and_failure(self):
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        errors = []

        def reset():
            try:
                cb.reset()
            except Exception as e:
                errors.append(e)

        def fail():
            try:
                cb.record_failure()
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=reset)
        t2 = threading.Thread(target=fail)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, f"Thread errors: {errors}"


# ============================================================================
# ErrorClassifier
# ============================================================================


class TestErrorClassifier:
    """Verify DeerFlow-compatible error classification with Chinese patterns."""

    # ── QUOTA ────────────────────────────────────────

    def test_quota_balance_chinese(self):
        err = ErrorClassifier.classify("账户余额不足")
        assert err.category == ErrorCategory.QUOTA
        assert not err.retryable

    def test_quota_exceeded_english(self):
        err = ErrorClassifier.classify("insufficient quota for model")
        assert err.category == ErrorCategory.QUOTA
        assert not err.retryable

    def test_quota_msg(self):
        err = ErrorClassifier.classify("您的额度已用完，请充值")
        assert err.category == ErrorCategory.QUOTA

    def test_quota_billing(self):
        err = ErrorClassifier.classify("billing account not active")
        assert err.category == ErrorCategory.QUOTA

    # ── AUTH ─────────────────────────────────────────

    def test_auth_chinese(self):
        err = ErrorClassifier.classify("API 鉴权失败，密钥无效")
        assert err.category == ErrorCategory.AUTH
        assert not err.retryable

    def test_auth_invalid_key(self):
        err = ErrorClassifier.classify("invalid api key provided")
        assert err.category == ErrorCategory.AUTH
        assert not err.retryable

    def test_auth_unauthorized(self):
        err = ErrorClassifier.classify("401 Unauthorized")
        assert err.category == ErrorCategory.AUTH

    # ── BUSY ─────────────────────────────────────────

    def test_busy_chinese(self):
        err = ErrorClassifier.classify("当前服务负载较高，请稍后再试")
        assert err.category == ErrorCategory.BUSY
        assert err.retryable

    def test_busy_server_busy(self):
        err = ErrorClassifier.classify("server is busy, please retry")
        assert err.category == ErrorCategory.BUSY
        assert err.retryable

    def test_busy_rate_limit(self):
        err = ErrorClassifier.classify("rate limit exceeded")
        assert err.category == ErrorCategory.BUSY
        assert err.retryable

    def test_busy_too_many_requests(self):
        err = ErrorClassifier.classify("请求过于频繁")
        assert err.category == ErrorCategory.BUSY

    # ── TRANSIENT ────────────────────────────────────

    def test_transient_timeout(self):
        err = ErrorClassifier.classify("connection timeout")
        assert err.category == ErrorCategory.TRANSIENT
        assert err.retryable

    def test_transient_network(self):
        err = ErrorClassifier.classify("网络连接失败")
        assert err.category == ErrorCategory.TRANSIENT
        assert err.retryable

    def test_transient_reset(self):
        err = ErrorClassifier.classify("Connection reset by peer")
        assert err.category == ErrorCategory.TRANSIENT

    def test_transient_unavailable(self):
        err = ErrorClassifier.classify("service unavailable")
        assert err.category == ErrorCategory.TRANSIENT

    # ── GENERIC ──────────────────────────────────────

    def test_generic_unknown(self):
        err = ErrorClassifier.classify("some random error we don't know about")
        assert err.category == ErrorCategory.GENERIC
        assert not err.retryable

    # ── HTTP status fallback ─────────────────────────

    def test_http_401_auth(self):
        err = ErrorClassifier.classify({"status": 401, "error": "Unauthorized"})
        assert err.category == ErrorCategory.AUTH

    def test_http_429_busy(self):
        err = ErrorClassifier.classify({"status_code": 429, "error": "Rate limited"})
        assert err.category == ErrorCategory.BUSY
        assert err.retryable

    def test_http_500_transient(self):
        err = ErrorClassifier.classify({"status": 500, "error": "Internal error"})
        assert err.category == ErrorCategory.TRANSIENT
        assert err.retryable

    def test_http_503_transient(self):
        err = ErrorClassifier.classify({"status_code": 503})
        assert err.category == ErrorCategory.TRANSIENT

    # ── Exception objects ────────────────────────────

    def test_classify_exception(self):
        err = ErrorClassifier.classify(Exception("Connection timeout"))
        assert err.category == ErrorCategory.TRANSIENT

    def test_classify_auth_exception(self):
        err = ErrorClassifier.classify(Exception("invalid api key"))
        assert err.category == ErrorCategory.AUTH

    # ── User messages ────────────────────────────────

    def test_user_message_quota(self):
        msg = ErrorClassifier.user_message(ErrorCategory.QUOTA)
        assert "余额" in msg or "配额" in msg  # Chinese

    def test_user_message_auth(self):
        msg = ErrorClassifier.user_message(ErrorCategory.AUTH)
        assert "密钥" in msg or "key" in msg.lower()

    def test_user_message_busy(self):
        msg = ErrorClassifier.user_message(ErrorCategory.BUSY)
        assert "重试" in msg

    def test_user_message_transient(self):
        msg = ErrorClassifier.user_message(ErrorCategory.TRANSIENT)
        assert "重试" in msg

    def test_user_message_generic(self):
        msg = ErrorClassifier.user_message(ErrorCategory.GENERIC)
        assert len(msg) > 0


# ============================================================================
# LLMRetryMiddleware integration
# ============================================================================


class TestLLMRetryWithClassifier:
    """Verify the retry middleware uses the new classifier + circuit breaker."""

    def test_circuit_breaker_created_by_default(self):
        from kairos.middleware.llm_retry import LLMRetryMiddleware
        mw = LLMRetryMiddleware()
        assert mw._circuit is not None
        assert mw._circuit.state == CircuitState.CLOSED

    def test_circuit_breaker_disabled(self):
        from kairos.middleware.llm_retry import LLMRetryMiddleware
        mw = LLMRetryMiddleware(enable_circuit_breaker=False)
        assert mw._circuit is None
        assert mw.circuit_state is None

    def test_circuit_state_property(self):
        from kairos.middleware.llm_retry import LLMRetryMiddleware
        mw = LLMRetryMiddleware()
        assert mw.circuit_state == CircuitState.CLOSED

    def test_user_friendly_error_integration(self):
        from kairos.middleware.llm_retry import LLMRetryMiddleware
        mw = LLMRetryMiddleware()
        # No error yet
        assert mw.user_friendly_error() is None

    def test_repr_includes_circuit_state(self):
        from kairos.middleware.llm_retry import LLMRetryMiddleware
        mw = LLMRetryMiddleware()
        assert "closed" in repr(mw)

        mw_no_cb = LLMRetryMiddleware(enable_circuit_breaker=False)
        assert "off" in repr(mw_no_cb)

    def test_classifier_custom(self):
        from kairos.middleware.llm_retry import LLMRetryMiddleware, ErrorClassifier
        custom = ErrorClassifier()
        mw = LLMRetryMiddleware(classifier=custom)
        assert mw._classifier is custom
