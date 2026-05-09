"""Browser tools — web scraping, screenshots, search, and form submission.

Tools:
  - web_scrape: fetch URL content and extract text, optionally filtered by CSS selector
  - web_screenshot: fetch URL and return raw HTML for LLM vision screenshot
  - web_search_advanced: enhanced web search via DuckDuckGo Lite
  - web_form_submit: submit HTML forms with GET or POST

All tools use stdlib only (urllib, html.parser, re).
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Any

from kairos.tools.registry import register_tool

USER_AGENT = (
    "Mozilla/5.0 (compatible; Kairos-Agent/1.0; +https://github.com/nousresearch/kairos)"
)
DEFAULT_TIMEOUT = 15


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
        # Add whitespace after block-level elements
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
    # Collapse whitespace
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


def _extract_title(html: str) -> str:
    """Extract <title> from HTML."""
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


# ── Simple CSS selector matching (tag, .class, #id, tag.class) ──────────────

def _matches_selector(tag: str, attrs: dict[str, str], selector: str) -> bool:
    """Check if an element matches a simple CSS selector.

    Supports: 'tag', '.class', '#id', 'tag.class', 'tag#id'.
    Multiple selectors can be comma-separated.
    """
    selectors = [s.strip() for s in selector.split(",") if s.strip()]
    for sel in selectors:
        if _match_single(tag, attrs, sel):
            return True
    return False


def _match_single(tag: str, attrs: dict[str, str], selector: str) -> bool:
    """Match a single simple selector."""
    s = selector
    # Extract #id
    id_part = ""
    m = re.search(r"#([\w-]+)", s)
    if m:
        id_part = m.group(1)
        s = s.replace("#" + id_part, "")

    # Extract classes
    classes = []
    for m in re.finditer(r"\.([\w-]+)", s):
        classes.append(m.group(1))
        s = s.replace("." + m.group(1), "", 1)

    # What remains is the tag name (or empty)
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
    """Extract text from elements matching a CSS selector."""

    def __init__(self, selector: str) -> None:
        super().__init__()
        self.selector = selector
        self.results: list[str] = []
        self._depth = 0
        self._matching: list[int] = []  # stack of match status per depth
        self._current_text: list[str] = []
        self._skip_tags = {"script", "style", "noscript"}

    def handle_starttag(self, tag: str, attrs_list: list[tuple[str, str | None]]) -> None:
        attrs = {k: (v or "") for k, v in attrs_list}
        if tag in self._skip_tags:
            self._matching.append(-1)  # skip mode
        elif any(v > 0 for v in self._matching):
            # Already inside a matching element — still collect text
            self._matching.append(1)
        elif _matches_selector(tag, attrs, self.selector):
            self._matching.append(2)  # start of match
            self._current_text = []
        else:
            self._matching.append(0)

        # Block-level newlines
        if tag in {"p", "div", "li", "br", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section", "article"}:
            if any(v >= 1 for v in self._matching):
                self._current_text.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if not self._matching:
            return
        state = self._matching.pop()
        if state == 2:
            # End of a top-level match
            text = "".join(self._current_text).strip()
            if text:
                self.results.append(text)
            self._current_text = []
        elif state == 1 and self._matching:
            # Still inside a match
            pass

    def handle_data(self, data: str) -> None:
        if self._matching and self._matching[-1] >= 1:
            s = data.strip()
            if s:
                self._current_text.append(s)
                self._current_text.append(" ")


def _extract_by_selector(html: str, selector: str) -> list[str]:
    """Extract text from elements matching a CSS selector."""
    extractor = _SelectorExtractor(selector)
    extractor.feed(html)
    extractor.close()
    return extractor.results


def _fetch_url(url: str, timeout: int = DEFAULT_TIMEOUT) -> tuple[int, str, dict[str, str]]:
    """Fetch a URL and return (status_code, body, headers_dict)."""
    if not url.startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")

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
    """Fetch URL and extract content, optionally filtered by CSS selector.

    Returns:
        dict with keys: url, title, status_code, content (or elements if selector used)
    """
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
    """Fetch a URL and return raw HTML for LLM vision screenshot analysis.

    This tool does not produce an actual image screenshot. Instead it returns
    the full HTML so that a vision-capable system can render it and analyze
    the visual layout.

    Returns:
        dict with keys: url, html, title, status_code, html_length
    """
    try:
        status, html, headers = _fetch_url(url)
        title = _extract_title(html)

        # Return HTML truncated to a reasonable size for LLM processing
        max_html = 200000
        truncated = len(html) > max_html
        html_out = html[:max_html]

        return {
            "url": url,
            "title": title,
            "status_code": status,
            "html": html_out,
            "html_length": len(html_out),
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
    description="Enhanced web search using DuckDuckGo Lite. Returns ranked results with title, URL, and snippet. No API key required.",
    parameters={
        "query": {"type": "string", "description": "Search query string"},
        "num_results": {"type": "integer", "description": "Number of results to return (default: 5, max: 20)"},
    },
    category="browser",
)
def web_search_advanced(query: str, num_results: int = 5) -> dict:
    """Search the web using DuckDuckGo Lite (no API key needed).

    Falls back to alternative parsing if results are sparse.

    Returns:
        dict with keys: query, results (list of {title, url, snippet}), source
    """
    num_results = max(1, min(num_results, 20))

    try:
        # Use DuckDuckGo Lite HTML interface
        params = urllib.parse.urlencode({"q": query})
        url = f"https://lite.duckduckgo.com/lite/?{params}"

        req = urllib.request.Request(url, headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html",
        })
        resp = urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT)
        html = resp.read().decode("utf-8", errors="replace")

        results = _parse_ddg_lite(html, num_results)

        # Fallback: if DDG Lite gave nothing, try the HTML version
        if not results:
            results = _parse_ddg_html(query, num_results)

        return {
            "query": query,
            "results": results,
            "count": len(results),
            "source": "duckduckgo",
            "query_url": url,
        }

    except urllib.error.HTTPError as e:
        return {"query": query, "results": [], "error": f"HTTP {e.code}: {e.reason}"}
    except Exception as e:
        return {"query": query, "results": [], "error": str(e)}


def _parse_ddg_lite(html: str, num_results: int) -> list[dict]:
    """Parse DuckDuckGo Lite HTML results.

    DDG Lite uses <a> tags for result links and <td> for snippets.
    """
    results: list[dict] = []

    # DDG Lite pattern: result links have class="result-link"
    # Snippet is in the next <td class="result-snippet">
    link_pattern = re.compile(
        r'<a[^>]*href="(https?://[^"]+)"[^>]*>(.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )
    snippet_pattern = re.compile(
        r'<td[^>]*class="result-snippet"[^>]*>(.*?)</td>',
        re.IGNORECASE | re.DOTALL,
    )

    links = link_pattern.findall(html)
    snippets = snippet_pattern.findall(html)

    # Also try simpler patterns
    if not links:
        # DDG Lite simple format: each result is a <tr> with link + snippet
        rows = re.split(r"<tr[^>]*>", html, flags=re.IGNORECASE)
        for row in rows:
            m = re.search(
                r'<a[^>]*href="(https?://[^"]+)"[^>]*>(.+?)</a>',
                row, re.IGNORECASE | re.DOTALL,
            )
            if m:
                url = m.group(1)
                # Skip duckduckgo.com internal links
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
                results.append({"title": title, "url": url, "snippet": snippet})
                if len(results) >= num_results:
                    break
        return results

    for i, (url, raw_title) in enumerate(links):
        if i >= num_results:
            break
        if "duckduckgo.com" in url:
            continue
        title = re.sub(r"<[^>]+>", "", raw_title).strip()
        snippet = ""
        if i < len(snippets):
            snippet = re.sub(r"<[^>]+>", " ", snippets[i]).strip()
            snippet = re.sub(r"\s+", " ", snippet)
        if url and title:
            results.append({"title": title, "url": url, "snippet": snippet})

    return results[:num_results]


def _parse_ddg_html(query: str, num_results: int) -> list[dict]:
    """Fallback: use DuckDuckGo HTML search."""
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
        # DDG HTML results are in elements with class "result"
        blocks = re.split(
            r'<div[^>]*class="[^"]*result[^"]*"[^>]*>',
            html, flags=re.IGNORECASE,
        )

        for block in blocks[1:]:
            if len(results) >= num_results:
                break
            # Extract URL and title
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
    """Submit a form and return the response.

    For GET requests, data is appended as query parameters.
    For POST requests, data is sent as application/x-www-form-urlencoded.

    Returns:
        dict with keys: url, method, status_code, headers, content, content_length
    """
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

        req = urllib.request.Request(
            final_url,
            data=encoded_data if method == "POST" else None,
            headers={
                "User-Agent": USER_AGENT,
                "Content-Type": "application/x-www-form-urlencoded",
            } if method == "POST" else {
                "User-Agent": USER_AGENT,
            },
        )
        # Override method for POST
        if method == "POST":
            req.method = "POST"

        resp = urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT)
        body = resp.read().decode("utf-8", errors="replace")
        headers = {k.lower(): v for k, v in resp.headers.items()}

        # Truncate response content
        max_content = 50000
        truncated = len(body) > max_content

        return {
            "url": final_url if method == "GET" else url,
            "method": method,
            "status_code": resp.status,
            "headers": headers,
            "content": body[:max_content],
            "content_length": len(body[:max_content]),
            "original_length": len(body),
            "truncated": truncated,
            "sent_data": form_data,
        }
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")[:10000]
        except Exception:
            pass
        return {
            "url": url, "method": method, "status_code": e.code,
            "headers": {k.lower(): v for k, v in e.headers.items()} if hasattr(e, "headers") else {},
            "content": body, "error": f"HTTP {e.code}: {e.reason}",
        }
    except urllib.error.URLError as e:
        return {"url": url, "method": method, "status_code": 0, "error": f"Connection error: {e.reason}"}
    except Exception as e:
        return {"url": url, "method": method, "status_code": 0, "error": str(e)}
