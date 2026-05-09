"""
Line Messaging API adapter.

Docs: https://developers.line.biz/en/docs/messaging-api/
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any

from kairos.gateway.protocol import (
    UnifiedMessage,
    UnifiedResponse,
    ConnectionState,
    MessageRole,
    ContentBlock,
)
from kairos.gateway.adapters.base import PlatformAdapter


class LineAdapter(PlatformAdapter):
    """Line Messaging API adapter — push + reply messages."""

    platform_name = "line"
    API_BASE = "https://api.line.me/v2"

    def __init__(self, channel_access_token: str = "", channel_secret: str = ""):
        super().__init__()
        self._token = channel_access_token
        self._secret = channel_secret

    async def connect(self) -> bool:
        if not self._token:
            self._state = ConnectionState.ERROR
            return False
        try:
            # Verify token
            url = f"{self.API_BASE}/bot/info"
            req = urllib.request.Request(
                url,
                headers={"Authorization": f"Bearer {self._token}"},
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
        """Send reply or push message to Line user."""
        if not self._token:
            return False

        try:
            url = f"{self.API_BASE}/bot/message/push"

            messages: list[dict[str, Any]] = []

            # Line text message (max 5000 chars)
            text = response.text[:5000]
            if text:
                messages.append({"type": "text", "text": text})

            # Image message if media URL
            if response.media_url:
                messages.append({
                    "type": "image",
                    "originalContentUrl": response.media_url,
                    "previewImageUrl": response.media_url,
                })

            if not messages:
                return False

            body = {
                "to": chat_id,
                "messages": messages,
            }

            data = json.dumps(body).encode()
            req = urllib.request.Request(
                url, data=data,
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Content-Type": "application/json",
                },
            )
            urllib.request.urlopen(req, timeout=10)
            return True
        except Exception:
            return False

    async def receive(self) -> UnifiedMessage | None:
        return None  # Webhook-based

    def translate_incoming(self, raw: dict[str, Any]) -> UnifiedMessage:
        """Parse Line webhook event."""
        events = raw.get("events", [raw] if isinstance(raw, dict) else [])
        if not events:
            return UnifiedMessage(
                id="empty", role=MessageRole.USER,
                content=[ContentBlock.text_block("")],
                platform="line", chat_id="",
            )

        event = events[0] if isinstance(events[0], dict) else {}

        event_type = event.get("type", "message")
        msg_type = ""
        content = ""

        if event_type == "message":
            message = event.get("message", {})
            msg_type = message.get("type", "text")

            if msg_type == "text":
                content = message.get("text", "")
            elif msg_type == "image":
                content = "[image]"
            elif msg_type == "video":
                content = "[video]"
            elif msg_type == "audio":
                content = "[audio]"
            elif msg_type == "location":
                content = "[location]"
            elif msg_type == "sticker":
                content = "[sticker]"
            else:
                content = f"[{msg_type}]"
        elif event_type == "postback":
            content = event.get("postback", {}).get("data", "")
        elif event_type == "follow":
            content = "[follow]"
        elif event_type == "unfollow":
            content = "[unfollow]"
        elif event_type == "join":
            content = "[join group]"
        else:
            content = f"[{event_type}]"

        source = event.get("source", {})

        return UnifiedMessage(
            id=event.get("webhookEventId", event.get("replyToken", "")),
            role=MessageRole.USER,
            content=[ContentBlock.text_block(content)],
            platform="line",
            chat_id=source.get("userId", source.get("groupId", source.get("roomId", ""))),
            thread_id=event.get("replyToken", ""),
            sender_id=source.get("userId", ""),
            sender_name="",
        )
