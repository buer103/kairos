"""
Discord Gateway adapter.

Uses Discord Bot Gateway + REST API (no external lib required).
Documents: https://discord.com/developers/docs/intro
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Any

from kairos.gateway.protocol import (
    UnifiedMessage,
    UnifiedResponse,
    ConnectionState,
    MessageRole,
    ContentBlock,
    ContentType,
)
from kairos.gateway.adapters.base import PlatformAdapter


class DiscordAdapter(PlatformAdapter):
    """Discord Bot adapter — REST API only (no real-time Gateway websocket).

    For production real-time, combine with discord.py.
    """

    platform_name = "discord"
    API_BASE = "https://discord.com/api/v10"

    def __init__(self, bot_token: str = ""):
        super().__init__()
        self._token = bot_token
        self._headers = {}
        if bot_token:
            self._headers = {
                "Authorization": f"Bot {bot_token}",
                "Content-Type": "application/json",
                "User-Agent": "Kairos/1.0",
            }

    async def connect(self) -> bool:
        if not self._token:
            self._state = ConnectionState.ERROR
            return False
        try:
            req = urllib.request.Request(
                f"{self.API_BASE}/users/@me",
                headers=self._headers,
            )
            resp = urllib.request.urlopen(req, timeout=10)
            if resp.status == 200:
                self._state = ConnectionState.CONNECTED
                return True
            self._state = ConnectionState.ERROR
            return False
        except Exception:
            self._state = ConnectionState.ERROR
            return False

    async def disconnect(self) -> None:
        self._state = ConnectionState.DISCONNECTED

    async def send(self, chat_id: str, response: UnifiedResponse) -> bool:
        """Send to a Discord channel or DM."""
        if not self._token:
            return False

        try:
            # Split content into <=2000 char chunks (Discord limit)
            chunks = _chunk_text(response.text, 1900)

            for i, chunk in enumerate(chunks):
                payload: dict[str, Any] = {"content": chunk}

                # Embed media if present
                if response.media_url and i == 0:
                    payload["embeds"] = [{
                        "image": {"url": response.media_url},
                    }]

                url = f"{self.API_BASE}/channels/{chat_id}/messages"
                data = json.dumps(payload).encode()
                req = urllib.request.Request(url, data=data, headers=self._headers, method="POST")
                urllib.request.urlopen(req, timeout=10)

            return True
        except Exception:
            return False

    async def receive(self) -> UnifiedMessage | None:
        return None  # Webhook/WebSocket-based; messages come via HTTP callback

    def translate_incoming(self, raw: dict[str, Any]) -> UnifiedMessage:
        """Parse a Discord interaction payload."""
        msg_type = raw.get("type", 1)

        # PING (verification)
        if msg_type == 1:
            return UnifiedMessage(
                id=raw.get("id", "ping"),
                role=MessageRole.SYSTEM,
                content=[ContentBlock.text_block("ping")],
                platform="discord",
                chat_id="",
                raw_payload={"type": 1},
            )

        # APPLICATION_COMMAND or MESSAGE_CREATE
        data = raw.get("data", raw)
        content = data.get("content", "")
        # Handle slash command options
        if isinstance(data, dict) and "options" in data:
            for opt in data["options"]:
                if opt.get("name") != "content":
                    content += f" /{opt['name']}:{opt.get('value', '')}"

        return UnifiedMessage(
            id=raw.get("id", raw.get("message_id", "")),
            role=MessageRole.USER,
            content=[ContentBlock.text_block(content)],
            platform="discord",
            chat_id=raw.get("channel_id", ""),
            thread_id=raw.get("guild_id", ""),
            sender_id=raw.get("author", {}).get("id", "") if isinstance(raw.get("author"), dict) else "",
            sender_name=raw.get("author", {}).get("username", "") if isinstance(raw.get("author"), dict) else "",
        )


def _chunk_text(text: str, max_len: int = 1900) -> list[str]:
    """Split text into chunks <= max_len, breaking at line boundaries."""
    if len(text) <= max_len:
        return [text]
    chunks = []
    while len(text) > max_len:
        split = text.rfind("\n", 0, max_len)
        if split == -1:
            split = text.rfind(" ", 0, max_len)
        if split == -1:
            split = max_len
        chunks.append(text[:split].strip())
        text = text[split:].strip()
    if text:
        chunks.append(text)
    return chunks
