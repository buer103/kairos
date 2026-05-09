"""View image middleware — injects image data for vision-capable models.

When the agent uses view_image tool and the model supports vision, this
middleware injects the image as base64 data into the message stream so
the model can "see" it.

DeerFlow layer 9 — runs before_model, injecting image data after tool completes.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from kairos.core.middleware import Middleware


class ViewImageMiddleware(Middleware):
    """Injects viewed images into the message stream for vision models.

    Hook: before_model — checks if a view_image tool just completed
    and injects the image as an image_url content block.

    Requires: model supports vision (set supports_vision=True).
    """

    def __init__(self, supports_vision: bool = False):
        self.supports_vision = supports_vision
        self._viewed_images: list[str] = []

    def before_model(self, state: Any, runtime: dict[str, Any]) -> dict[str, Any] | None:
        if not self.supports_vision:
            return None

        messages = getattr(state, "messages", [])
        if not messages:
            return None

        # Check if the last assistant message called view_image
        last_ai_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant" and messages[i].get("tool_calls"):
                last_ai_idx = i
                break

        if last_ai_idx < 0:
            return None

        ai_msg = messages[last_ai_idx]
        view_calls = [
            tc for tc in ai_msg.get("tool_calls", [])
            if tc.get("function", {}).get("name") == "view_image"
        ]
        if not view_calls:
            return None

        # Check if all view_image calls have completed
        tool_ids = {tc["id"] for tc in view_calls}
        completed = set()
        for m in messages[last_ai_idx + 1:]:
            if m.get("role") == "tool" and m.get("tool_call_id") in tool_ids:
                completed.add(m["tool_call_id"])

        if tool_ids != completed:
            return None  # Not all view_image calls done yet

        # Find the actual image paths from tool results
        image_paths = []
        for m in messages[last_ai_idx + 1:]:
            if m.get("role") == "tool" and m.get("tool_call_id") in tool_ids:
                content = m.get("content", "")
                if isinstance(content, str) and content.startswith("/"):
                    image_paths.append(content)

        if not image_paths:
            return None

        # Inject image content
        for img_path in image_paths:
            try:
                data = self._encode_image(img_path)
                if data:
                    ext = Path(img_path).suffix.lower().lstrip(".")
                    mime = f"image/{ext}" if ext in ("png", "jpeg", "jpg", "gif", "webp") else "image/png"
                    image_block = {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime};base64,{data}",
                                    "detail": "high",
                                },
                            }
                        ],
                    }
                    state.messages.append(image_block)
            except Exception:
                pass

        return None

    @staticmethod
    def _encode_image(path: str) -> str | None:
        """Read and base64-encode an image file."""
        try:
            p = Path(path)
            if not p.exists() or p.stat().st_size > 20 * 1024 * 1024:  # 20 MB limit
                return None
            return base64.b64encode(p.read_bytes()).decode("utf-8")
        except Exception:
            return None

    def __repr__(self) -> str:
        return f"ViewImageMiddleware(supports_vision={self.supports_vision})"
