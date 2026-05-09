"""View image middleware — image injection for vision models with caching + dedup.

DeerFlow layer 9 — holds viewed images across turns, avoids re-encoding.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

from kairos.core.middleware import Middleware

logger = logging.getLogger("kairos.middleware.view_image")

MIME_ALIASES = {
    "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
    "gif": "image/gif", "webp": "image/webp", "bmp": "image/bmp",
}

MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20 MB
MAX_IMAGES_PER_TURN = 10
CACHE_MAX = 50


class ViewImageMiddleware(Middleware):
    """Injects viewed images into message stream for vision models.

    Caches base64 encodings to avoid re-reading the same image.
    Tracks viewed images across turns for dedup.
    """

    def __init__(self, supports_vision: bool = False):
        self.supports_vision = supports_vision
        self._viewed: set[str] = set()  # Dedup across turns
        self._cache: dict[str, str] = {}  # path → base64

    def before_model(self, state: Any, runtime: dict[str, Any]) -> dict[str, Any] | None:
        if not self.supports_vision:
            return None

        messages = getattr(state, "messages", [])
        if not messages:
            return None

        # Find assistant messages with pending view_image tool calls
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") != "assistant":
                continue
            tcs = messages[i].get("tool_calls", [])
            view_calls = [tc for tc in tcs if tc.get("function", {}).get("name") == "view_image"]
            if not view_calls:
                continue

            # Get completed tool results
            call_ids = {tc["id"] for tc in view_calls}
            completed = set()
            image_paths = []
            for m in messages[i + 1:]:
                if m.get("role") == "tool" and m.get("tool_call_id") in call_ids:
                    completed.add(m["tool_call_id"])
                    content = m.get("content", "")
                    if isinstance(content, str) and content.startswith("/"):
                        image_paths.append(content)

            if call_ids != completed:
                return None  # Not done yet

            if not image_paths:
                return None

            # Inject images (deduped)
            injected = 0
            for img_path in image_paths:
                if img_path in self._viewed:
                    continue
                if injected >= MAX_IMAGES_PER_TURN:
                    break

                data = self._encode(img_path)
                if not data:
                    continue

                ext = Path(img_path).suffix.lower().lstrip(".")
                mime = MIME_ALIASES.get(ext, "image/png")

                image_block = {
                    "role": "user",
                    "content": [{
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{data}",
                            "detail": "high",
                        },
                    }],
                }
                messages.append(image_block)
                self._viewed.add(img_path)
                injected += 1

            if injected:
                logger.debug("Injected %d images", injected)
            break

        return None

    def after_agent(self, state: Any, runtime: dict[str, Any]) -> dict[str, Any] | None:
        """Clear dedup set for next session."""
        self._viewed.clear()
        # Trim cache
        if len(self._cache) > CACHE_MAX:
            oldest = sorted(self._cache.keys())[:len(self._cache) - CACHE_MAX]
            for k in oldest:
                del self._cache[k]
        return None

    def _encode(self, path: str) -> str | None:
        if path in self._cache:
            return self._cache[path]
        try:
            p = Path(path)
            if not p.exists() or p.stat().st_size > MAX_IMAGE_SIZE:
                return None
            data = base64.b64encode(p.read_bytes()).decode()
            if len(self._cache) < CACHE_MAX:
                self._cache[path] = data
            return data
        except Exception:
            return None

    def __repr__(self) -> str:
        return f"ViewImageMiddleware(vision={self.supports_vision}, cache={len(self._cache)})"
