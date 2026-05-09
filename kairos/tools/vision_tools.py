"""Vision tools — image analysis helpers for LLM vision capabilities.

Tools:
  - vision_analyze: read and base64-encode an image for LLM vision analysis
  - vision_compare: side-by-side comparison of two images
  - vision_screenshot_analyze: render HTML for visual analysis by the agent

These tools prepare image data in a format suitable for multi-modal LLM
messages. They do not perform analysis themselves; they package the data
so the Agent Loop can inject it as a vision message.
"""

from __future__ import annotations

import base64
import os
import re
from pathlib import Path
from typing import Any

from kairos.tools.registry import register_tool

# Common image MIME types by extension
_MIME_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".svg": "image/svg+xml",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".ico": "image/x-icon",
}

# Max image size for base64 encoding (20 MB)
_MAX_IMAGE_SIZE = 20 * 1024 * 1024


def _detect_mime(filepath: str) -> str:
    """Detect MIME type from file extension or magic bytes."""
    ext = Path(filepath).suffix.lower()
    if ext in _MIME_MAP:
        return _MIME_MAP[ext]

    # Try magic bytes for common formats
    try:
        with open(filepath, "rb") as f:
            header = f.read(12)
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if header.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if header.startswith(b"GIF87a") or header.startswith(b"GIF89a"):
            return "image/gif"
        if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
            return "image/webp"
        if header.startswith(b"BM"):
            return "image/bmp"
        if header.startswith(b"<svg") or header.startswith(b"<?xml"):
            return "image/svg+xml"
    except Exception:
        pass

    return "application/octet-stream"


def _read_and_encode(filepath: str) -> tuple[str, str, int]:
    """Read an image file and return (base64_data, mime_type, byte_size).

    Returns empty base64 string on error.
    """
    p = Path(filepath).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {filepath}")
    if p.stat().st_size > _MAX_IMAGE_SIZE:
        raise ValueError(f"Image too large: {p.stat().st_size} bytes (max {_MAX_IMAGE_SIZE})")

    data = p.read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    mime = _detect_mime(str(p))
    return encoded, mime, len(data)


def _clean_html_for_rendering(html: str) -> str:
    """Basic HTML cleanup to make it more renderable.

    - Ensures it has basic structure (html/head/body tags)
    - Removes scripts for safety
    - Extracts or builds a self-contained snippet
    """
    # Strip script tags
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.IGNORECASE | re.DOTALL)

    # If it already has <html>, return as-is
    if re.search(r"<html", html, re.IGNORECASE):
        return html

    # Wrap in basic structure
    title = ""
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if m:
        title = m.group(1).strip()

    # Extract body content if present
    body_content = html
    m = re.search(r"<body[^>]*>(.*?)</body>", html, re.IGNORECASE | re.DOTALL)
    if m:
        body_content = m.group(1)

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>{title}</title></head>
<body>{body_content}</body>
</html>"""


# ── Tools ────────────────────────────────────────────────────────────────────

@register_tool(
    name="vision_analyze",
    description="Read an image file and prepare it for LLM vision analysis. Returns base64-encoded image data with MIME type.",
    parameters={
        "image_path": {"type": "string", "description": "Absolute or relative path to the image file"},
        "question": {"type": "string", "description": "Optional question about the image to guide analysis"},
    },
    category="vision",
)
def vision_analyze(image_path: str, question: str = "") -> dict:
    """Prepare an image for vision-capable LLM analysis.

    Reads the image file, base64-encodes it, and returns structured data
    that can be injected into a multi-modal LLM message as an image_url
    content part.

    Returns:
        dict with keys: image_data (base64), mime_type, file_path, file_size,
                        question, image_url (data URI)
    """
    try:
        b64_data, mime_type, byte_size = _read_and_encode(image_path)
        p = Path(image_path).expanduser().resolve()

        data_uri = f"data:{mime_type};base64,{b64_data}"

        result: dict[str, Any] = {
            "file_path": str(p),
            "file_name": p.name,
            "mime_type": mime_type,
            "file_size": byte_size,
            "image_data": b64_data,
            "image_url": data_uri,
        }

        if question:
            result["question"] = question

        return result

    except FileNotFoundError as e:
        return {"error": str(e), "image_path": image_path}
    except ValueError as e:
        return {"error": str(e), "image_path": image_path}
    except Exception as e:
        return {"error": f"Failed to read image: {e}", "image_path": image_path}


@register_tool(
    name="vision_compare",
    description="Prepare two images for side-by-side comparison by a vision-capable LLM.",
    parameters={
        "image_a": {"type": "string", "description": "Path to the first image file"},
        "image_b": {"type": "string", "description": "Path to the second image file"},
        "question": {"type": "string", "description": "Optional question about the comparison"},
    },
    category="vision",
)
def vision_compare(image_a: str, image_b: str, question: str = "") -> dict:
    """Prepare two images for side-by-side vision comparison.

    Both images are base64-encoded and returned as data URIs.
    The Agent Loop can present them together in a multi-modal message.

    Returns:
        dict with keys: images (list of {path, mime_type, size, data}) and optional question
    """
    images: list[dict] = []
    errors: list[str] = []

    for label, path in [("image_a", image_a), ("image_b", image_b)]:
        try:
            b64_data, mime_type, byte_size = _read_and_encode(path)
            p = Path(path).expanduser().resolve()
            images.append({
                "label": label,
                "file_path": str(p),
                "file_name": p.name,
                "mime_type": mime_type,
                "file_size": byte_size,
                "image_data": b64_data,
                "image_url": f"data:{mime_type};base64,{b64_data}",
            })
        except (FileNotFoundError, ValueError) as e:
            errors.append(f"{label}: {e}")
        except Exception as e:
            errors.append(f"{label}: {e}")

    result: dict[str, Any] = {
        "images": images,
        "count": len(images),
    }

    if errors:
        result["errors"] = errors
    if question:
        result["question"] = question
    if len(images) == 2:
        result["image_a"] = images[0]
        result["image_b"] = images[1]

    return result


@register_tool(
    name="vision_screenshot_analyze",
    description="Prepare an HTML string for visual rendering and analysis. Returns metadata to guide the Agent in rendering the HTML as a screenshot for vision analysis.",
    parameters={
        "html": {"type": "string", "description": "Raw HTML content to render and analyze"},
        "question": {"type": "string", "description": "Optional question to guide the visual analysis of the rendered HTML"},
    },
    category="vision",
)
def vision_screenshot_analyze(html: str, question: str = "") -> dict:
    """Prepare HTML content for visual rendering analysis.

    This tool does not render HTML itself. Instead, it cleans the HTML,
    extracts metadata, and provides guidance so the Agent Loop can render
    it (e.g., by writing to a temp .html file and taking a screenshot) and
    then analyze the visual result with a vision-capable model.

    Returns:
        dict with keys: html_length, cleaned_html, title, meta_description,
                        element_count, question, rendering_instructions
    """
    try:
        cleaned = _clean_html_for_rendering(html)
        html_len = len(cleaned)

        # Extract metadata
        title = ""
        m = re.search(r"<title[^>]*>(.*?)</title>", cleaned, re.IGNORECASE | re.DOTALL)
        if m:
            title = m.group(1).strip()

        meta_desc = ""
        m = re.search(
            r'<meta[^>]*name="description"[^>]*content="([^"]*)"',
            cleaned, re.IGNORECASE,
        )
        if m:
            meta_desc = m.group(1)

        # Count HTML elements (rough)
        element_count = len(re.findall(r"<\w+", cleaned))

        # Estimate rendering dimensions from common patterns
        viewport_width = "auto"
        m = re.search(r'<meta[^>]*name="viewport"[^>]*content="([^"]*)"', cleaned, re.IGNORECASE)
        if m:
            viewport_width = m.group(1)

        result: dict[str, Any] = {
            "html_length": html_len,
            "truncated": html_len < len(html),
            "cleaned_html": cleaned[:200000],  # Truncate for transport
            "title": title,
            "meta_description": meta_desc,
            "element_count": element_count,
            "viewport": viewport_width,
            "has_styles": bool(re.search(r"<style|<link.*stylesheet", cleaned, re.IGNORECASE)),
            "has_scripts": bool(re.search(r"<script", html, re.IGNORECASE)),
            "rendering_instructions": (
                "To analyze this page visually: save the cleaned_html to a temporary "
                ".html file, open it in a browser, take a screenshot, and pass the "
                "screenshot to vision_analyze with your question. "
                f"Page title: '{title}'. "
                f"Total elements: {element_count}."
            ),
        }

        if question:
            result["question"] = question

        return result

    except Exception as e:
        return {"error": str(e), "html_length": len(html) if html else 0}
