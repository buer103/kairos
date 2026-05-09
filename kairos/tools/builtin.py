"""Built-in tools — essential tools every Kairos agent needs.

Tools:
  - read_file: read a file with line numbers and pagination
  - write_file: create or overwrite a file
  - terminal: execute shell commands
  - web_search: search the web (stub — requires API key)
  - web_fetch: fetch URL content
  - list_files: list directory contents
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from kairos.tools.registry import register_tool


@register_tool(
    name="read_file",
    description="Read a file with line numbers. Use offset and limit for large files.",
    parameters={
        "path": {"type": "string", "description": "Absolute or relative path to the file"},
        "offset": {"type": "integer", "description": "Line number to start from (1-indexed, default: 1)"},
        "limit": {"type": "integer", "description": "Max lines to read (default: 500, max: 2000)"},
    },
)
def read_file(path: str, offset: int = 1, limit: int = 500) -> dict:
    """Read a file with line numbers."""
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return {"error": f"File not found: {path}"}
        if p.is_dir():
            return {"error": f"Path is a directory: {path}"}
        if p.stat().st_size > 50 * 1024 * 1024:
            return {"error": "File too large (>50 MB). Use offset/limit to read specific sections."}

        lines = p.read_text(encoding="utf-8", errors="replace").split("\n")
        total = len(lines)
        limit = max(1, min(limit, 2000))
        offset = max(1, min(offset, total))

        end = min(offset + limit - 1, total)
        result_lines = []
        for i in range(offset - 1, end):
            result_lines.append(f"{i + 1:6d}|{lines[i]}")

        return {
            "content": "\n".join(result_lines),
            "total_lines": total,
            "start_line": offset,
            "end_line": end,
            "path": str(p),
        }
    except Exception as e:
        return {"error": str(e)}


@register_tool(
    name="write_file",
    description="Create or overwrite a file with the given content. Creates parent directories automatically.",
    parameters={
        "path": {"type": "string", "description": "Path to the file to write (absolute or relative)"},
        "content": {"type": "string", "description": "Content to write to the file"},
    },
)
def write_file(path: str, content: str) -> dict:
    """Write content to a file, creating parent directories."""
    try:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)

        original = p.read_text(encoding="utf-8") if p.exists() else None
        p.write_text(content, encoding="utf-8")

        return {
            "path": str(p),
            "bytes_written": len(content.encode("utf-8")),
            "overwritten": original is not None,
        }
    except Exception as e:
        return {"error": str(e)}


@register_tool(
    name="terminal",
    description="Execute a shell command. Returns stdout, stderr, and exit code. Timeout: 120s.",
    parameters={
        "command": {"type": "string", "description": "Shell command to execute"},
        "workdir": {"type": "string", "description": "Working directory (optional)"},
        "timeout": {"type": "integer", "description": "Timeout in seconds (default: 120, max: 300)"},
    },
)
def terminal(command: str, workdir: str = "", timeout: int = 120) -> dict:
    """Execute a shell command."""
    try:
        timeout = max(1, min(timeout, 300))
        cwd = str(Path(workdir).expanduser().resolve()) if workdir else None

        start = time.time()
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        elapsed = (time.time() - start) * 1000

        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"

        # Truncate long output
        max_output = 50000
        if len(output) > max_output:
            output = output[:max_output] + f"\n... (truncated, total {len(output)} chars)"

        return {
            "stdout": result.stdout[:max_output],
            "stderr": result.stderr[:10000] if result.stderr else "",
            "exit_code": result.returncode,
            "duration_ms": round(elapsed, 1),
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"Command timed out after {timeout}s", "exit_code": -1}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "exit_code": -1}


@register_tool(
    name="list_files",
    description="List files in a directory. Supports glob patterns.",
    parameters={
        "path": {"type": "string", "description": "Directory path (default: current directory)"},
        "pattern": {"type": "string", "description": "Glob pattern to filter files (e.g., '*.py', default: '*')"},
        "limit": {"type": "integer", "description": "Max files to return (default: 50, max: 200)"},
    },
)
def list_files(path: str = ".", pattern: str = "*", limit: int = 50) -> dict:
    """List files in a directory."""
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return {"error": f"Directory not found: {path}"}
        if not p.is_dir():
            return {"error": f"Not a directory: {path}"}

        limit = max(1, min(limit, 200))
        entries = []
        for f in sorted(p.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True):
            if limit <= 0:
                break
            if f.name.startswith(".") and pattern == "*":
                continue
            try:
                st = f.stat()
                entries.append({
                    "name": f.name,
                    "path": str(f),
                    "type": "directory" if f.is_dir() else "file",
                    "size": st.st_size,
                    "modified": st.st_mtime,
                })
                limit -= 1
            except OSError:
                continue

        return {"path": str(p), "pattern": pattern, "entries": entries, "total": len(entries)}
    except Exception as e:
        return {"error": str(e)}


@register_tool(
    name="web_search",
    description="Search the web for information. Returns titles, URLs, and snippets.",
    parameters={
        "query": {"type": "string", "description": "Search query"},
        "num_results": {"type": "integer", "description": "Number of results (default: 5, max: 10)"},
    },
)
def web_search(query: str, num_results: int = 5) -> dict:
    """Search the web (stub — configure a search API for production use).

    To enable real search, set one of:
      - SERPER_API_KEY (https://serper.dev)
      - TAVILY_API_KEY (https://tavily.com)
      - BRAVE_API_KEY (https://brave.com/search/api/)
    """
    import urllib.request
    import urllib.parse

    api_key = os.environ.get("SERPER_API_KEY") or os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return {
            "query": query,
            "results": [],
            "error": "No search API key configured. Set SERPER_API_KEY, TAVILY_API_KEY, or BRAVE_API_KEY.",
        }

    num_results = max(1, min(num_results, 10))

    # Try Serper (Google Search API)
    if os.environ.get("SERPER_API_KEY"):
        try:
            url = "https://google.serper.dev/search"
            data = json.dumps({"q": query, "num": num_results}).encode()
            req = urllib.request.Request(url, data=data, headers={
                "X-API-KEY": api_key,
                "Content-Type": "application/json",
            })
            resp = urllib.request.urlopen(req, timeout=10)
            result = json.loads(resp.read().decode())

            return {
                "query": query,
                "results": [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("link", ""),
                        "snippet": r.get("snippet", ""),
                    }
                    for r in result.get("organic", [])[:num_results]
                ],
            }
        except Exception as e:
            return {"query": query, "results": [], "error": f"Search failed: {e}"}

    # Try Tavily
    if os.environ.get("TAVILY_API_KEY"):
        try:
            url = "https://api.tavily.com/search"
            data = json.dumps({
                "api_key": api_key,
                "query": query,
                "max_results": num_results,
            }).encode()
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            resp = urllib.request.urlopen(req, timeout=10)
            result = json.loads(resp.read().decode())

            return {
                "query": query,
                "results": [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "snippet": r.get("content", ""),
                    }
                    for r in result.get("results", [])[:num_results]
                ],
            }
        except Exception as e:
            return {"query": query, "results": [], "error": f"Search failed: {e}"}

    return {"query": query, "results": [], "error": "Search not configured."}


@register_tool(
    name="web_fetch",
    description="Fetch the content of a URL. Returns text content.",
    parameters={
        "url": {"type": "string", "description": "URL to fetch (must start with http:// or https://)"},
        "max_chars": {"type": "integer", "description": "Max characters to return (default: 10000, max: 50000)"},
    },
)
def web_fetch(url: str, max_chars: int = 10000) -> dict:
    """Fetch URL content."""
    import urllib.request
    import urllib.error

    if not url.startswith(("http://", "https://")):
        return {"error": "URL must start with http:// or https://"}

    max_chars = min(max_chars, 50000)

    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Kairos-Agent/1.0",
        })
        resp = urllib.request.urlopen(req, timeout=10)
        content = resp.read().decode("utf-8", errors="replace")

        # Strip HTML tags for plain text (simple approach)
        import re
        text = re.sub(r"<[^>]+>", " ", content)
        text = re.sub(r"\s+", " ", text).strip()
        text = text[:max_chars]

        return {
            "url": url,
            "content": text,
            "status_code": resp.status,
            "content_type": resp.headers.get("Content-Type", ""),
            "truncated": len(content) > max_chars,
        }
    except urllib.error.HTTPError as e:
        return {"url": url, "content": "", "status_code": e.code, "error": f"HTTP {e.code}"}
    except Exception as e:
        return {"url": url, "content": "", "status_code": 0, "error": str(e)}
