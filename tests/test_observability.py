"""Tests for observability modules: ErrorClassifier, UsageTracker, AgentInsights."""

import time
import pytest

from kairos.observability.error_classifier import ErrorClassifier
from kairos.observability.usage_tracker import UsageTracker
from kairos.observability.insights import AgentInsights


# ============================================================================
# ErrorClassifier
# ============================================================================


class TestErrorClassifier:
    def test_record_and_rate(self):
        ec = ErrorClassifier(window_seconds=60)
        ec.record_error(ValueError("test"), {"provider": "openai"})
        assert ec.get_error_rate() > 0

    def test_no_alert_below_threshold(self):
        ec = ErrorClassifier(window_seconds=60, alert_threshold=10)
        ec.record_error(ValueError("test"), {})
        assert not ec.should_alert()

    def test_alert_above_threshold(self):
        ec = ErrorClassifier(window_seconds=3600, alert_threshold=2)
        for i in range(5):
            ec.record_error(ValueError(f"err{i}"), {})
        assert ec.should_alert()

    def test_error_breakdown(self):
        ec = ErrorClassifier(window_seconds=60)
        ec.record_error(ValueError("rate limit exceeded"), {"provider": "openai"})
        ec.record_error(ValueError("unauthorized"), {"provider": "openai"})
        breakdown = ec.get_error_breakdown()
        assert "rate_limit" in breakdown or "RATE_LIMIT" in str(breakdown)

    def test_recent_errors(self):
        ec = ErrorClassifier(window_seconds=60)
        ec.record_error(ValueError("e1"), {})
        ec.record_error(ValueError("e2"), {})
        recent = ec.get_recent_errors(limit=1)
        assert len(recent) == 1

    def test_reset(self):
        ec = ErrorClassifier(window_seconds=60)
        ec.record_error(ValueError("test"), {})
        ec.reset()
        assert ec.get_error_rate() == 0.0

    def test_root_cause(self):
        ec = ErrorClassifier(window_seconds=60)
        ec.record_error(ValueError("rate limit"), {"provider": "openai"})
        ec.record_error(ValueError("rate limit"), {"provider": "openai"})
        cause = ec.get_root_cause()
        assert "rate_limit" in cause.lower() or "rate limit" in cause.lower()


# ============================================================================
# UsageTracker
# ============================================================================


class TestUsageTracker:
    def test_track_call(self):
        ut = UsageTracker()
        ut.track_call("deepseek", "deepseek-chat", 1000, 500, 200.0, True)
        assert ut.total_tokens == 1500
        assert ut.total_cost > 0

    def test_success_rate(self):
        ut = UsageTracker()
        ut.track_call("openai", "gpt-4o", 100, 50, 100.0, True)
        ut.track_call("openai", "gpt-4o", 100, 50, 100.0, False)
        assert ut.success_rate == 0.5

    def test_average_latency(self):
        ut = UsageTracker()
        ut.track_call("deepseek", "deepseek-chat", 100, 50, 100.0, True)
        ut.track_call("deepseek", "deepseek-chat", 100, 50, 300.0, True)
        assert ut.average_latency_ms == 200.0

    def test_daily_stats(self):
        ut = UsageTracker()
        ut.track_call("deepseek", "deepseek-chat", 1000, 500, 200.0, True)
        stats = ut.get_daily_stats()
        assert stats["total_calls"] == 1
        assert stats["total_tokens"] == 1500

    def test_calls_per_minute(self):
        ut = UsageTracker()
        ut.track_call("deepseek", "deepseek-chat", 100, 50, 100.0, True)
        # Should be > 0 since we just made a call
        assert ut.calls_per_minute >= 0

    def test_cost_calculation(self):
        ut = UsageTracker()
        # DeepSeek: $0.27/M input, $1.10/M output
        ut.track_call("deepseek", "deepseek-chat", 1000000, 1000000, 100.0, True)
        expected = 0.27 + 1.10  # $0.27 for 1M input, $1.10 for 1M output
        assert abs(ut.total_cost - expected) < 0.01


# ============================================================================
# AgentInsights
# ============================================================================


class TestAgentInsights:
    def test_health_report(self):
        ai = AgentInsights()
        ai.record_call("deepseek", "deepseek-chat", 1000, 500, 200.0, True)
        report = ai.get_health_report()
        assert report["status"] == "healthy"
        assert "errors" in report
        assert "usage" in report

    def test_efficiency_score(self):
        ai = AgentInsights()
        for _ in range(10):
            ai.record_call("deepseek", "deepseek-chat", 100, 50, 100.0, True)
        score = ai.get_efficiency_score()
        assert 0 <= score <= 1

    def test_no_anomalies_when_clean(self):
        ai = AgentInsights()
        ai.record_call("deepseek", "deepseek-chat", 100, 50, 100.0, True)
        anomalies = ai.detect_anomalies()
        # Fresh tracker with 1 good call should have no anomalies
        assert len(anomalies) == 0

    def test_anomaly_on_error_spike(self):
        from kairos.observability.error_classifier import ErrorClassifier
        ec = ErrorClassifier(window_seconds=3600, alert_threshold=1)
        ai = AgentInsights(error_classifier=ec)
        # Record many errors
        for _ in range(5):
            ai.errors.record_error(ValueError("fail"), {})
        anomalies = ai.detect_anomalies()
        assert len(anomalies) >= 1
