"""Tests for observability metrics — Prometheus-compatible registry."""

from __future__ import annotations

import pytest

from kairos.observability.metrics import (
    MetricsRegistry,
    _Counter,
    _Gauge,
    _Histogram,
    get_metrics,
    reset_metrics,
)


# ============================================================================
# Counter
# ============================================================================


class TestCounter:
    def test_inc(self):
        c = _Counter("test")
        c.inc()
        assert c.get() == 1.0

    def test_inc_amount(self):
        c = _Counter("test")
        c.inc(5.0)
        assert c.get() == 5.0

    def test_thread_safety(self):
        import threading
        c = _Counter("test")
        def add():
            for _ in range(100):
                c.inc()
        threads = [threading.Thread(target=add) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert c.get() == 500.0


# ============================================================================
# Gauge
# ============================================================================


class TestGauge:
    def test_set(self):
        g = _Gauge("test")
        g.set(42.0)
        assert g.get() == 42.0

    def test_inc_dec(self):
        g = _Gauge("test")
        g.inc(5)
        g.dec(2)
        assert g.get() == 3.0


# ============================================================================
# Histogram
# ============================================================================


class TestHistogram:
    def test_observe(self):
        h = _Histogram("test", buckets=[0.5, 1.0, 2.0])
        h.observe(0.3)
        h.observe(1.5)
        h.observe(0.7)

        assert h.get_count() == 3
        assert h.get_sum() == pytest.approx(2.5)
        assert h.get_buckets()[0.5] == 1
        assert h.get_buckets()[1.0] == 2  # 0.3 and 0.7 both <= 1.0
        assert h.get_buckets()[2.0] == 3  # all <= 2.0

    def test_large_value(self):
        h = _Histogram("test", buckets=[1.0, 5.0])
        h.observe(100.0)
        # Value > all buckets, only counted in total
        assert h.get_count() == 1
        assert h.get_buckets()[1.0] == 0
        assert h.get_buckets()[5.0] == 0


# ============================================================================
# MetricsRegistry — Prometheus format
# ============================================================================


class TestMetricsRegistry:
    @pytest.fixture(autouse=True)
    def _reset(self):
        reset_metrics()
        yield

    def test_counter_inc_and_render(self):
        reg = MetricsRegistry()
        reg.inc("requests_total", labels={"status": "success"})
        reg.inc("requests_total", labels={"status": "success"})
        reg.inc("requests_total", labels={"status": "error"})

        output = reg.render()
        assert 'kairos_requests_total{status="success"}' in output
        assert " 2" in output[output.find("success"): output.find("success") + 20]
        assert 'kairos_requests_total{status="error"}' in output

    def test_histogram_render(self):
        reg = MetricsRegistry()
        reg.observe("request_duration_seconds", 0.1)
        reg.observe("request_duration_seconds", 0.5)
        reg.observe("request_duration_seconds", 2.0)

        output = reg.render()
        assert "# HELP" in output
        assert "# TYPE kairos_request_duration_seconds histogram" in output
        assert "kairos_request_duration_seconds_sum" in output
        assert "kairos_request_duration_seconds_count" in output
        assert "le=" in output

    def test_gauge_render(self):
        reg = MetricsRegistry()
        reg.set_gauge("active_sessions", 3.0)
        output = reg.render()
        assert "kairos_active_sessions" in output
        assert "gauge" in output

    def test_update_process_metrics(self):
        reg = MetricsRegistry()
        reg.update_process_metrics()
        output = reg.render()
        assert "uptime_seconds" in output
        assert "metrics_count_total" in output

    def test_singleton(self):
        reg1 = get_metrics()
        reg2 = get_metrics()
        assert reg1 is reg2

    def test_render_empty(self):
        reg = MetricsRegistry()
        assert isinstance(reg.render(), str)

    def test_label_formatting(self):
        reg = MetricsRegistry()
        reg.inc("test", labels={"env": "prod", "region": "us-east"})
        output = reg.render()
        assert "env" in output
        assert "region" in output
        assert "prod" in output

    def test_name_prefix(self):
        reg = MetricsRegistry(name_prefix="kairos")
        reg.inc("tool_calls_total")
        output = reg.render()
        assert "kairos_tool_calls_total" in output

    def test_auto_prefix(self):
        reg = MetricsRegistry(name_prefix="kairos")
        reg.inc("kairos_test")
        output = reg.render()
        # Should not double-prefix
        assert "kairos_kairos" not in output


# ============================================================================
# MetricsRegistry — concurrent access
# ============================================================================


class TestMetricsConcurrency:
    @pytest.fixture(autouse=True)
    def _reset(self):
        reset_metrics()
        yield

    def test_concurrent_inc(self):
        import threading
        reg = get_metrics()

        def inc():
            for _ in range(100):
                reg.inc("requests_total")

        threads = [threading.Thread(target=inc) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        output = reg.render()
        assert "1000" in output  # 10 * 100
