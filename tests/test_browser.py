"""Tests for Playwright browser backend — graceful fallback when not installed."""

from __future__ import annotations

import pytest

from kairos.browser import PlaywrightBrowser


class TestPlaywrightBrowser:
    def test_constructor_does_not_import_playwright(self):
        """Constructor should not fail if Playwright not installed."""
        browser = PlaywrightBrowser()
        # available may be True or False depending on env
        assert isinstance(browser.available, bool)

    def test_fetch_graceful_fallback(self):
        """fetch() returns error dict when Playwright unavailable."""
        browser = PlaywrightBrowser()
        if browser.available:
            pytest.skip("Playwright is installed, skipping fallback test")
        result = browser.fetch("https://example.com")
        assert "error" in result
        assert "url" in result

    def test_screenshot_graceful_fallback(self):
        browser = PlaywrightBrowser()
        if browser.available:
            pytest.skip("Playwright installed")
        result = browser.screenshot("https://example.com")
        assert "error" in result

    def test_extract_links_graceful_fallback(self):
        browser = PlaywrightBrowser()
        if browser.available:
            pytest.skip("Playwright installed")
        result = browser.extract_links("https://example.com")
        assert "error" in result

    def test_interact_graceful_fallback(self):
        browser = PlaywrightBrowser()
        if browser.available:
            pytest.skip("Playwright installed")
        result = browser.interact("https://example.com", [
            {"action": "click", "selector": "#btn"}
        ])
        assert "error" in result

    def test_start_without_playwright_raises(self):
        browser = PlaywrightBrowser()
        if browser.available:
            pytest.skip("Playwright installed")
        with pytest.raises(RuntimeError, match="not installed"):
            browser.start()

    def test_stop_idempotent(self):
        """stop() should not raise on unstarted browser."""
        browser = PlaywrightBrowser()
        browser.stop()  # no error
        browser.close()  # no error

    def test_context_manager_not_started(self):
        """Close before start is safe."""
        browser = PlaywrightBrowser()
        browser.close()
