"""Headless browser backend — Playwright-powered web interaction.

Optional dependency: playwright (pip install playwright && playwright install chromium)

Usage:
    from kairos.browser.playwright_backend import PlaywrightBrowser
    browser = PlaywrightBrowser()
    html = browser.fetch("https://example.com")
    screenshot = browser.screenshot("https://example.com")
    browser.close()
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger("kairos.browser")

USER_AGENT = (
    "Mozilla/5.0 (compatible; Kairos-Agent/1.0; +https://github.com/buer103/kairos)"
)


class PlaywrightBrowser:
    """Headless browser using Playwright for JavaScript-rendered content.

    Requires: pip install playwright && playwright install chromium
    Falls back gracefully if Playwright is not installed.
    """

    def __init__(self, headless: bool = True, timeout: int = 30000):
        self._headless = headless
        self._timeout = timeout
        self._browser = None
        self._context = None
        self._playwright = None
        self._available = False

        try:
            from playwright.sync_api import sync_playwright
            self._playwright_api = sync_playwright
            self._available = True
        except ImportError:
            logger.info("Playwright not installed. Run: pip install playwright && playwright install chromium")
            self._playwright_api = None

    @property
    def available(self) -> bool:
        return self._available

    def start(self) -> None:
        """Launch the browser."""
        if not self._available:
            raise RuntimeError("Playwright not installed")
        if self._browser:
            return  # already running

        self._playwright = self._playwright_api().start()
        self._browser = self._playwright.chromium.launch(headless=self._headless)
        self._context = self._browser.new_context(
            user_agent=USER_AGENT,
            viewport={"width": 1280, "height": 720},
        )
        logger.info("Playwright browser started (headless=%s)", self._headless)

    def stop(self) -> None:
        """Close the browser."""
        if self._context:
            self._context.close()
            self._context = None
        if self._browser:
            self._browser.close()
            self._browser = None
        if self._playwright:
            self._playwright.stop()
            self._playwright = None

    def close(self) -> None:
        """Alias for stop()."""
        self.stop()

    def fetch(
        self,
        url: str,
        wait_for_selector: str | None = None,
        wait_ms: int = 0,
        extract_text: bool = True,
    ) -> dict[str, Any]:
        """Fetch and render a page with full JavaScript execution.

        Returns:
            dict with keys: url, status, title, html, text, headers
        """
        if not self._available:
            return {"error": "Playwright not installed", "url": url}

        started = self._browser is not None
        if not started:
            self.start()

        try:
            page = self._context.new_page()
            page.set_default_timeout(self._timeout)

            response = page.goto(url, wait_until="domcontentloaded")

            if wait_for_selector:
                page.wait_for_selector(wait_for_selector, timeout=self._timeout)
            if wait_ms:
                page.wait_for_timeout(wait_ms)

            result: dict[str, Any] = {
                "url": page.url,
                "status": response.status if response else 0,
                "title": page.title(),
                "html": page.content(),
            }

            if extract_text:
                result["text"] = page.inner_text("body")

            # Headers
            if response:
                result["headers"] = dict(response.headers)

            return result
        except Exception as e:
            logger.warning("Playwright fetch error: %s", e)
            return {"error": str(e), "url": url}
        finally:
            page.close()
            if not started:
                self.stop()

    def screenshot(
        self,
        url: str,
        full_page: bool = True,
        output_path: str | None = None,
    ) -> dict[str, Any]:
        """Take a screenshot of a page.

        Returns:
            dict with keys: url, screenshot (base64 if no output_path), path, size_bytes
        """
        if not self._available:
            return {"error": "Playwright not installed", "url": url}

        started = self._browser is not None
        if not started:
            self.start()

        try:
            page = self._context.new_page()
            page.set_default_timeout(self._timeout)
            page.goto(url, wait_until="domcontentloaded")

            screenshot_bytes = page.screenshot(full_page=full_page)

            result: dict[str, Any] = {
                "url": url,
                "size_bytes": len(screenshot_bytes),
            }

            if output_path:
                with open(output_path, "wb") as f:
                    f.write(screenshot_bytes)
                result["path"] = output_path
            else:
                import base64
                result["screenshot"] = base64.b64encode(screenshot_bytes).decode()

            return result
        except Exception as e:
            logger.warning("Playwright screenshot error: %s", e)
            return {"error": str(e), "url": url}
        finally:
            page.close()
            if not started:
                self.stop()

    def extract_links(self, url: str) -> dict[str, Any]:
        """Extract all links from a page."""
        if not self._available:
            return {"error": "Playwright not installed", "url": url}

        started = self._browser is not None
        if not started:
            self.start()

        try:
            page = self._context.new_page()
            page.set_default_timeout(self._timeout)
            page.goto(url, wait_until="domcontentloaded")

            links = page.evaluate("""() => {
                return Array.from(document.querySelectorAll('a[href]')).map(a => ({
                    text: a.textContent.trim().substring(0, 200),
                    href: a.href
                }));
            }""")

            return {"url": url, "links": links, "count": len(links)}
        except Exception as e:
            return {"error": str(e), "url": url}
        finally:
            page.close()
            if not started:
                self.stop()

    def interact(
        self,
        url: str,
        actions: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Perform a sequence of interactions on a page.

        Each action is a dict with:
            - action: "click", "fill", "select", "press", "wait"
            - selector: CSS selector
            - value: text to fill, key to press, or ms to wait

        Returns page state after all actions.
        """
        if not self._available:
            return {"error": "Playwright not installed", "url": url}

        started = self._browser is not None
        if not started:
            self.start()

        try:
            page = self._context.new_page()
            page.set_default_timeout(self._timeout)
            page.goto(url, wait_until="domcontentloaded")

            for action in actions:
                act = action.get("action", "")
                sel = action.get("selector", "")
                val = action.get("value", "")

                if act == "click" and sel:
                    page.click(sel)
                elif act == "fill" and sel:
                    page.fill(sel, val)
                elif act == "select" and sel:
                    page.select_option(sel, val)
                elif act == "press" and val:
                    page.keyboard.press(val)
                elif act == "wait":
                    page.wait_for_timeout(int(val) if val.isdigit() else 1000)

            return {
                "url": page.url,
                "title": page.title(),
                "text": page.inner_text("body"),
                "actions_performed": len(actions),
            }
        except Exception as e:
            return {"error": str(e), "url": url}
        finally:
            page.close()
            if not started:
                self.stop()
