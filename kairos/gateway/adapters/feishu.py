"""
Feishu (Lark) adapter.

Uses Feishu Open API with tenant_access_token auto-refresh.
Docs: https://open.feishu.cn/document/home/index
"""

from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from typing import Any

from kairos.gateway.protocol import (
    UnifiedMessage,
    UnifiedResponse,
    ConnectionState,
    MessageRole,
    ContentBlock,
)
from kairos.gateway.adapters.base import PlatformAdapter


class FeishuAdapter(PlatformAdapter):
    """Feishu bot adapter — IM message send/receive via Open API."""

    platform_name = "feishu"
    API_BASE = "https://open.feishu.cn/open-apis"

    def __init__(self, app_id: str = "", app_secret: str = ""):
        super().__init__()
        self._app_id = app_id
        self._app_secret = app_secret
        self._tenant_token: str = ""
        self._token_expiry: float = 0

    async def connect(self) -> bool:
        if not self._app_id or not self._app_secret:
            self._state = ConnectionState.ERROR
            return False
        return await self._refresh_token()

    async def disconnect(self) -> None:
        self._state = ConnectionState.DISCONNECTED

    async def _refresh_token(self) -> bool:
        """Get or refresh tenant_access_token (valid 2 hours)."""
        if self._tenant_token and time.time() < self._token_expiry - 60:
            return True

        try:
            url = f"{self.API_BASE}/auth/v3/tenant_access_token/internal"
            data = json.dumps({
                "app_id": self._app_id,
                "app_secret": self._app_secret,
            }).encode()
            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json; charset=utf-8"},
            )
            resp = urllib.request.urlopen(req, timeout=10)
            result = json.loads(resp.read().decode())
            if result.get("code") == 0:
                self._tenant_token = result["tenant_access_token"]
                self._token_expiry = time.time() + (result.get("expire", 7200) - 60)
                self._state = ConnectionState.CONNECTED
                return True
            self._state = ConnectionState.ERROR
            return False
        except Exception:
            self._state = ConnectionState.ERROR
            return False

    async def send(self, chat_id: str, response: UnifiedResponse) -> bool:
        """Send message via Feishu IM API."""
        if not await self._refresh_token():
            return False

        try:
            url = f"{self.API_BASE}/im/v1/messages?receive_id_type=chat_id"
            body: dict[str, Any] = {
                "receive_id": chat_id,
                "msg_type": "interactive" if response.media_url else "text",
                "content": "",
            }

            if response.media_url:
                body["content"] = json.dumps({
                    "elements": [{
                        "tag": "markdown",
                        "content": response.text,
                    }],
                    "header": {"title": {"tag": "plain_text", "content": "Kairos"}},
                }, ensure_ascii=False)
            else:
                body["content"] = json.dumps({"text": response.text}, ensure_ascii=False)

            data = json.dumps(body, ensure_ascii=False).encode()
            req = urllib.request.Request(
                url, data=data,
                headers={
                    "Authorization": f"Bearer {self._tenant_token}",
                    "Content-Type": "application/json; charset=utf-8",
                },
            )
            resp = urllib.request.urlopen(req, timeout=10)
            result = json.loads(resp.read().decode())
            return result.get("code") == 0
        except Exception:
            return False

    async def receive(self) -> UnifiedMessage | None:
        return None  # Event subscription via webhook

    def translate_incoming(self, raw: dict[str, Any]) -> UnifiedMessage:
        """Parse Feishu event payload."""
        # Handle URL verification challenge
        if raw.get("type") == "url_verification":
            return UnifiedMessage(
                id=raw.get("uuid", "verify"),
                role=MessageRole.SYSTEM,
                content=[ContentBlock.text_block(raw.get("challenge", ""))],
                platform="feishu",
                chat_id="",
                raw_payload={"challenge": raw.get("challenge", "")},
            )

        event = raw.get("event", raw)
        header = raw.get("header", {})

        msg_content = ""
        if isinstance(event, dict):
            message = event.get("message", event)
            if isinstance(message, dict):
                content = message.get("content", "{}")
                try:
                    parsed = json.loads(content)
                    msg_content = parsed.get("text", "")
                except (json.JSONDecodeError, TypeError):
                    msg_content = content
            msg_content = msg_content or event.get("text", "")

        return UnifiedMessage(
            id=header.get("event_id", raw.get("uuid", "")),
            role=MessageRole.USER,
            content=[ContentBlock.text_block(msg_content)],
            platform="feishu",
            chat_id=event.get("chat_id", "") if isinstance(event, dict) else "",
            thread_id=header.get("event_type", ""),
            sender_id=header.get("sender_id", ""),
        )
