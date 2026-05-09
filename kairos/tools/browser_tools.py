"""Browser tools — web scraping, search, screenshot, form submission.

All tools use stdlib only (urllib, html.parser, re).
Proxy support via environment variables (HTTP_PROXY, HTTPS_PROXY).
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Any

from kairos.tools.registry import register_tool

USER_AGENT = (
    "Mozilla/5.0 (compatible; Kairos-Agent/1.0; +https://github.com/buer103/kairos)"
)
DEFAULT_TIMEOUT = 15


# ── Proxy configuration ────────────────────────────────────────────────────


def _setup_proxy() -> dict[str, str] | None:
    """Detect and return proxy configuration from environment.

    Checks in order:
      1. HTTPS_PROXY / https_proxy
      2. HTTP_PROXY / http_proxy
      3. ALL_PROXY / all_proxy

    Returns dict suitable for urllib.request.ProxyHandler, or None if no proxy.
    """
    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        proxy = os.environ.get(var, "")
        if proxy:
            return {"https": proxy, "http": proxy}
    return None


def _build_opener(timeout: int = DEFAULT_TIMEOUT) -> urllib.request.OpenerDirector:
    """Build a URL opener with proxy support and timeout."""
    handlers: list[Any] = []

    # Proxy handler
    proxy_config = _setup_proxy()
    if proxy_config:
        handlers.append(urllib.request.ProxyHandler(proxy_config))

    # Install opener globally
    opener = urllib.request.build_opener(*handlers) if handlers else urllib.request.build_opener()
    urllib.request.install_opener(opener)
    return opener


# ── HTML text extraction helpers ────────────────────────────────────────────


class _TextExtractor(HTMLParser):
    """Extract visible text from HTML, skipping script/style tags."""

    def __init__(self) -> None:
        super().__init__()
        self.text_parts: list[str] = []
        self._skip = False
        self._skip_tags = {"script", "style", "noscript", "head"}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._skip_tags:
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in self._skip_tags:
            self._skip = False
        if tag in {"p", "div", "li", "br", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section", "article"}:
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            text = data.strip()
            if text:
                self.text_parts.append(text)


def _extract_text(html: str) -> str:
    """Extract visible text from HTML."""
    extractor = _TextExtractor()
    extractor.feed(html)
    extractor.close()
    raw = " ".join(extractor.text_parts)
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


def _extract_title(html: str) -> str:
    """Extract <title> from HTML."""
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


# ── Simple CSS selector matching ────────────────────────────────────────────


def _matches_selector(tag: str, attrs: dict[str, str], selector: str) -> bool:
    selectors = [s.strip() for s in selector.split(",") if s.strip()]
    for sel in selectors:
        if _match_single(tag, attrs, sel):
            return True
    return False


def _match_single(tag: str, attrs: dict[str, str], selector: str) -> bool:
    s = selector
    id_part = ""
    m = re.search(r"#([\w-]+)", s)
    if m:
        id_part = m.group(1)
        s = s.replace("#" + id_part, "")
    classes = []
    for m in re.finditer(r"\.([\w-]+)", s):
        classes.append(m.group(1))
        s = s.replace("." + m.group(1), "", 1)
    tag_part = s.strip()
    if tag_part and tag_part.lower() != tag.lower():
        return False
    if id_part and attrs.get("id") != id_part:
        return False
    if classes:
        elem_classes = set(attrs.get("class", "").split())
        if not set(classes).issubset(elem_classes):
            return False
    return True


class _SelectorExtractor(HTMLParser):
    def __init__(self, selector: str) -> None:
        super().__init__()
        self.selector = selector
        self.results: list[str] = []
        self._matching: list[int] = []
        self._current_text: list[str] = []
        self._skip_tags = {"script", "style", "noscript"}

    def handle_starttag(self, tag: str, attrs_list: list[tuple[str, str | None]]) -> None:
        attrs = {k: (v or "") for k, v in attrs_list}
        if tag in self._skip_tags:
            self._matching.append(-1)
        elif any(v > 0 for v in self._matching):
            self._matching.append(1)
        elif _matches_selector(tag, attrs, self.selector):
            self._matching.append(2)
            self._current_text = []
        else:
            self._matching.append(0)
        if tag in {"p", "div", "li", "br", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section", "article"}:
            if any(v >= 1 for v in self._matching):
                self._current_text.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if not self._matching:
            return
        state = self._matching.pop()
        if state == 2:
            text = "".join(self._current_text).strip()
            if text:
                self.results.append(text)
            self._current_text = []

    def handle_data(self, data: str) -> None:
        if self._matching and self._matching[-1] >= 1:
            s = data.strip()
            if s:
                self._current_text.append(s)
                self._current_text.append(" ")


def _extract_by_selector(html: str, selector: str) -> list[str]:
    extractor = _SelectorExtractor(selector)
    extractor.feed(html)
    extractor.close()
    return extractor.results


def _fetch_url(url: str, timeout: int = DEFAULT_TIMEOUT) -> tuple[int, str, dict[str, str]]:
    """Fetch a URL and return (status_code, body, headers_dict)."""
    if not url.startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")

    # Build opener with proxy support
    _build_opener(timeout)

    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    })
    resp = urllib.request.urlopen(req, timeout=timeout)
    body = resp.read().decode("utf-8", errors="replace")
    headers = {k.lower(): v for k, v in resp.headers.items()}
    return resp.status, body, headers


# ── Tools ────────────────────────────────────────────────────────────────────


@register_tool(
    name="web_scrape",
    description="Fetch URL content and extract text. If a CSS selector is given, only matching elements are returned. Supports basic selectors: tag, .class, #id, tag.class.",
    parameters={
        "url": {"type": "string", "description": "URL to fetch (must start with http:// or https://)"},
        "selector": {"type": "string", "description": "Optional CSS selector to filter elements (e.g., 'h1', '.article', '#main', 'a.link')"},
    },
    category="browser",
)
def web_scrape(url: str, selector: str = "") -> dict:
    try:
        status, html, headers = _fetch_url(url)
        title = _extract_title(html)

        if selector:
            elements = _extract_by_selector(html, selector)
            return {
                "url": url,
                "title": title,
                "status_code": status,
                "selector": selector,
                "elements": elements,
                "count": len(elements),
            }

        text = _extract_text(html)
        return {
            "url": url,
            "title": title,
            "status_code": status,
            "content": text[:50000],
            "content_length": len(text),
            "truncated": len(text) > 50000,
        }
    except urllib.error.HTTPError as e:
        return {"url": url, "title": "", "status_code": e.code, "error": f"HTTP {e.code}: {e.reason}"}
    except urllib.error.URLError as e:
        return {"url": url, "title": "", "status_code": 0, "error": f"Connection error: {e.reason}"}
    except Exception as e:
        return {"url": url, "title": "", "status_code": 0, "error": str(e)}


@register_tool(
    name="web_screenshot",
    description="Fetch a URL and return the raw HTML content. The HTML can be rendered and analyzed by a vision-capable LLM.",
    parameters={
        "url": {"type": "string", "description": "URL to fetch (must start with http:// or https://)"},
    },
    category="browser",
)
def web_screenshot(url: str) -> dict:
    try:
        status, html, headers = _fetch_url(url)
        title = _extract_title(html)
        max_html = 200000
        truncated = len(html) > max_html
        return {
            "url": url,
            "title": title,
            "status_code": status,
            "html": html[:max_html],
            "html_length": min(len(html), max_html),
            "original_length": len(html),
            "truncated": truncated,
            "content_type": headers.get("content-type", ""),
        }
    except urllib.error.HTTPError as e:
        return {"url": url, "title": "", "status_code": e.code, "html": "", "error": f"HTTP {e.code}: {e.reason}"}
    except urllib.error.URLError as e:
        return {"url": url, "title": "", "status_code": 0, "html": "", "error": f"Connection error: {e.reason}"}
    except Exception as e:
        return {"url": url, "title": "", "status_code": 0, "html": "", "error": str(e)}


@register_tool(
    name="web_search_advanced",
    description="Enhanced web search using DuckDuckGo Lite + fallback engines. Returns ranked results with title, URL, and snippet. No API key required. Supports proxy via HTTPS_PROXY env.",
    parameters={
        "query": {"type": "string", "description": "Search query string"},
        "num_results": {"type": "integer", "description": "Number of results to return (default: 5, max: 20)"},
        "engine": {"type": "string", "description": "Search engine: 'duckduckgo' (default), 'brave', 'google_scholar', 'wikipedia'"},
    },
    category="browser",
)
def web_search_advanced(query: str, num_results: int = 5, engine: str = "duckduckgo") -> dict:
    """Search the web using configurable engine with proxy support.

    Engines:
      - duckduckgo: DuckDuckGo Lite HTML (no API key, most private)
      - brave: Brave Search (requires BRAVE_API_KEY env)
      - google_scholar: Google Scholar scraping
      - wikipedia: Wikipedia API (no key, most reliable)
    """
    num_results = max(1, min(num_results, 20))
    _build_opener(DEFAULT_TIMEOUT)

    engines = {
        "duckduckgo": _search_ddg,
        "brave": _search_brave,
        "google_scholar": _search_scholar,
        "wikipedia": _search_wikipedia,
    }

    searcher = engines.get(engine, _search_ddg)
    try:
        results = searcher(query, num_results)
        return {
            "query": query,
            "results": results,
            "count": len(results),
            "source": engine,
        }
    except Exception as e:
        # Fallback to Wikipedia if primary engine fails
        if engine != "wikipedia":
            try:
                results = _search_wikipedia(query, num_results)
                if results:
                    return {"query": query, "results": results, "count": len(results), "source": "wikipedia_fallback"}
            except Exception:
                pass
        return {"query": query, "results": [], "error": str(e)}


def _search_ddg(query: str, num_results: int) -> list[dict]:
    """DuckDuckGo Lite search."""
    params = urllib.parse.urlencode({"q": query})
    url = f"https://lite.duckduckgo.com/lite/?{params}"

    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "text/html",
    })
    resp = urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT)
    html = resp.read().decode("utf-8", errors="replace")

    results = _parse_ddg_lite(html, num_results)
    if not results:
        results = _parse_ddg_html(query, num_results)
    return results


def _search_brave(query: str, num_results: int) -> list[dict]:
    """Brave Search API (requires BRAVE_API_KEY)."""
    api_key = os.environ.get("BRAVE_API_KEY", "")
    if not api_key:
        raise ValueError("BRAVE_API_KEY not set in environment")

    params = urllib.parse.urlencode({"q": query, "count": min(num_results, 20)})
    url = f"https://api.search.brave.com/res/v1/web/search?{params}"

    req = urllib.request.Request(url, headers={
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    })
    resp = urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT)
    data = json.loads(resp.read().decode("utf-8"))

    results: list[dict] = []
    for result in data.get("web", {}).get("results", [])[:num_results]:
        results.append({
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "snippet": result.get("description", ""),
        })
    return results


def _search_scholar(query: str, num_results: int) -> list[dict]:
    """Google Scholar scraping (fragile — use sparingly)."""
    params = urllib.parse.urlencode({"q": query})
    url = f"https://scholar.google.com/scholar?{params}&hl=en"

    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "text/html",
    })
    resp = urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT)
    html = resp.read().decode("utf-8", errors="replace")

    results: list[dict] = []
    # Parse Scholar results — each result starts with <h3 class="gs_rt">
    blocks = re.split(r'<h3 class="gs_rt">', html)
    for block in blocks[1:num_results + 1]:
        m = re.search(r'<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.DOTALL)
        if m:
            title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
            snippet_match = re.search(r'<div class="gs_rs">(.*?)</div>', block, re.DOTALL)
            snippet = re.sub(r"<[^>]+>", " ", snippet_match.group(1)).strip() if snippet_match else ""
            results.append({"title": title, "url": m.group(1), "snippet": snippet})
    return results


def _search_wikipedia(query: str, num_results: int) -> list[dict]:
    """Wikipedia search API (no API key needed, most reliable)."""
    params = urllib.parse.urlencode({
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": min(num_results, 20),
    })
    url = f"https://en.wikipedia.org/w/api.php?{params}"

    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    })
    resp = urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT)
    data = json.loads(resp.read().decode("utf-8"))

    results: list[dict] = []
    for item in data.get("query", {}).get("search", [])[:num_results]:
        title = item.get("title", "")
        page_id = item.get("pageid", "")
        results.append({
            "title": title,
            "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
            "snippet": item.get("snippet", ""),
            "page_id": page_id,
        })
    return results


def _parse_ddg_lite(html: str, num_results: int) -> list[dict]:
    """Parse DuckDuckGo Lite HTML results."""
    results: list[dict] = []
    rows = re.split(r"<tr[^>]*>", html, flags=re.IGNORECASE)
    for row in rows:
        if len(results) >= num_results:
            break
        m = re.search(
            r'<a[^>]*href="(https?://[^"]+)"[^>]*>(.+?)</a>',
            row, re.IGNORECASE | re.DOTALL,
        )
        if m:
            url = m.group(1)
            if "duckduckgo.com" in url:
                continue
            title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
            snippet = ""
            s = re.search(
                r'<td[^>]*class="result-snippet"[^>]*>(.*?)</td>',
                row, re.IGNORECASE | re.DOTALL,
            )
            if s:
                snippet = re.sub(r"<[^>]+>", " ", s.group(1)).strip()
                snippet = re.sub(r"\s+", " ", snippet)
            if url and title:
                results.append({"title": title, "url": url, "snippet": snippet})
    return results


def _parse_ddg_html(query: str, num_results: int) -> list[dict]:
    """Fallback: DuckDuckGo HTML search."""
    try:
        params = urllib.parse.urlencode({"q": query})
        url = f"https://html.duckduckgo.com/html/?{params}"
        req = urllib.request.Request(url, headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html",
        })
        resp = urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT)
        html = resp.read().decode("utf-8", errors="replace")

        results: list[dict] = []
        blocks = re.split(
            r'<div[^>]*class="[^"]*result[^"]*"[^>]*>',
            html, flags=re.IGNORECASE,
        )
        for block in blocks[1:]:
            if len(results) >= num_results:
                break
            m = re.search(
                r'<a[^>]*class="result__a"[^>]*href="(https?://[^"]+)"[^>]*>(.*?)</a>',
                block, re.IGNORECASE | re.DOTALL,
            )
            if not m:
                m = re.search(
                    r'<a[^>]*href="(https?://[^"]+)"[^>]*>(.*?)</a>',
                    block, re.IGNORECASE | re.DOTALL,
                )
            if m:
                url = m.group(1)
                title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
                snippet = ""
                s = re.search(
                    r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
                    block, re.IGNORECASE | re.DOTALL,
                )
                if s:
                    snippet = re.sub(r"<[^>]+>", " ", s.group(1)).strip()
                if url and title and "duckduckgo.com" not in url:
                    results.append({"title": title, "url": url, "snippet": snippet})
        return results
    except Exception:
        return []


@register_tool(
    name="web_form_submit",
    description="Submit an HTML form to a URL using GET or POST with optional data fields.",
    parameters={
        "url": {"type": "string", "description": "Form action URL (must start with http:// or https://)"},
        "method": {"type": "string", "description": "HTTP method: 'GET' or 'POST' (default: 'GET')"},
        "data": {"type": "string", "description": "Form data as JSON string of key-value pairs (e.g., '{\"q\": \"hello\"}'). Default: empty object."},
    },
    category="browser",
)
def web_form_submit(url: str, method: str = "GET", data: str = "{}") -> dict:
    method = method.upper().strip()
    if method not in ("GET", "POST"):
        return {"url": url, "method": method, "status_code": 0, "error": "Method must be GET or POST"}

    try:
        form_data: dict[str, str] = json.loads(data) if data else {}
    except json.JSONDecodeError:
        return {"url": url, "method": method, "status_code": 0, "error": f"Invalid JSON data: {data}"}

    try:
        encoded_data = urllib.parse.urlencode(form_data).encode("utf-8") if form_data else b""
        final_url = url
        if method == "GET" and encoded_data:
            separator = "&" if "?" in url else "?"
            final_url = url + separator + encoded_data.decode("utf-8")

        _build_opener(DEFAULT_TIMEOUT)
        req = urllib.request.Request(
            final_url,
            data=encoded_data if method == "POST" else None,
            headers={
                "User-Agent": USER_AGENT,
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method=method,
        )
        resp = urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT)
        body = resp.read().decode("utf-8", errors="replace")
        headers = {k.lower(): v for k, v in resp.headers.items()}

        result: dict[str, Any] = {
            "url": final_url,
            "method": method,
            "status_code": resp.status,
            "headers": {k: v for k, v in list(headers.items())[:10]},
            "content_length": len(body),
        }
        if "text/html" in headers.get("content-type", ""):
            result["content"] = body[:5000]
        else:
            result["content"] = body[:5000]
        return result
    except urllib.error.HTTPError as e:
        return {"url": url, "method": method, "status_code": e.code, "error": f"HTTP {e.code}: {e.reason}"}
    except urllib.error.URLError as e:
        return {"url": url, "method": method, "status_code": 0, "error": f"Connection error: {e.reason}"}
    except Exception as e:
        return {"url": url, "method": method, "status_code": 0, "error": str(e)}
