"""
Signal adapter.

Uses signal-cli REST API (https://github.com/bbernhard/signal-cli-rest-api).
signal-cli provides a JSON-RPC REST wrapper around the Signal protocol.
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


class SignalAdapter(PlatformAdapter):
    """Signal adapter via signal-cli REST API.

    Requires a running signal-cli-rest-api container or service.
    """

    platform_name = "signal"
    DEFAULT_BASE = "http://localhost:8080"

    def __init__(self, base_url: str = "", sender_number: str = ""):
        """
        Args:
            base_url: signal-cli REST API base URL (default: http://localhost:8080)
            sender_number: Your Signal phone number (e.g., +1234567890)
        """
        super().__init__()
        self._base = base_url or self.DEFAULT_BASE
        self._sender = sender_number

    async def connect(self) -> bool:
        if not self._sender:
            self._state = ConnectionState.ERROR
            return False
        try:
            # Health check
            url = f"{self._base}/v1/about"
            resp = urllib.request.urlopen(url, timeout=5)
            if resp.status == 200:
                self._state = ConnectionState.CONNECTED
                return True
            self._state = ConnectionState.ERROR
            return False
        except Exception:
            # Signal adapter works even if health check fails
            self._state = ConnectionState.CONNECTED
            return True

    async def disconnect(self) -> None:
        self._state = ConnectionState.DISCONNECTED

    async def send(self, chat_id: str, response: UnifiedResponse) -> bool:
        """Send message via signal-cli REST API."""
        try:
            url = f"{self._base}/v2/send"

            body: dict[str, Any] = {
                "number": self._sender,
                "recipients": [chat_id],
                "message": response.text,
            }

            # Attach media if present
            if response.media_url:
                body["base64_attachments"] = [response.media_url]

            data = json.dumps(body).encode()
            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=15)
            return True
        except Exception:
            return False

    async def receive(self) -> UnifiedMessage | None:
        """Poll for new messages via signal-cli REST API."""
        try:
            url = f"{self._base}/v1/receive/{self._sender}"
            req = urllib.request.Request(url)
            resp = urllib.request.urlopen(req, timeout=5)
            messages = json.loads(resp.read().decode())
            if messages and isinstance(messages, list) and messages:
                return self.translate_incoming(messages[0])
            return None
        except Exception:
            return None

    def translate_incoming(self, raw: dict[str, Any]) -> UnifiedMessage:
        """Parse a signal-cli message."""
        envelope = raw.get("envelope", raw)
        data_message = envelope.get("dataMessage", {})

        content = data_message.get("message", "")
        attachments = data_message.get("attachments", [])
        if attachments:
            content += f" [attachment: {attachments[0].get('contentType', 'unknown')}]"

        return UnifiedMessage(
            id=raw.get("timestamp", ""),
            role=MessageRole.USER,
            content=[ContentBlock.text_block(content)],
            platform="signal",
            chat_id=envelope.get("source", ""),
            thread_id=data_message.get("groupInfo", {}).get("groupId", ""),
            sender_id=envelope.get("source", ""),
            sender_name=envelope.get("sourceName", ""),
        )
