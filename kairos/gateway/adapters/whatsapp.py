"""
WhatsApp Cloud API adapter.

Uses Meta's WhatsApp Cloud API (graph.facebook.com).
Docs: https://developers.facebook.com/docs/whatsapp/cloud-api
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


class WhatsAppAdapter(PlatformAdapter):
    """WhatsApp Cloud API adapter — send/receive via Meta Business API."""

    platform_name = "whatsapp"
    API_BASE = "https://graph.facebook.com/v18.0"

    def __init__(self, phone_number_id: str = "", access_token: str = ""):
        """
        Args:
            phone_number_id: WhatsApp Business phone number ID
            access_token: Meta permanent access token
        """
        super().__init__()
        self._phone_id = phone_number_id
        self._token = access_token

    async def connect(self) -> bool:
        if not self._phone_id or not self._token:
            self._state = ConnectionState.ERROR
            return False
        try:
            # Verify connection by checking phone number info
            url = f"{self.API_BASE}/{self._phone_id}"
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
        """Send text + optional media to a WhatsApp chat."""
        if not self._token:
            return False

        try:
            url = f"{self.API_BASE}/{self._phone_id}/messages"

            body: dict[str, Any] = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": chat_id,
                "type": "text",
                "text": {"body": response.text[:4096]},  # WhatsApp 4096 char limit
            }

            # If there's media, send as image/document
            if response.media_url:
                # Send as image with caption
                body["type"] = "image"
                body["image"] = {
                    "link": response.media_url,
                    "caption": response.text[:1024],
                }
                del body["text"]

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
        """Parse WhatsApp webhook payload."""
        # Meta verification challenge
        if "hub.challenge" in str(raw) or raw.get("hub", {}).get("challenge"):
            return UnifiedMessage(
                id="verify",
                role=MessageRole.SYSTEM,
                content=[ContentBlock.text_block(str(raw.get("hub", {}).get("challenge", raw.get("challenge", ""))))],
                platform="whatsapp",
                chat_id="",
                raw_payload=raw,
            )

        # Extract message from webhook payload
        entry = raw
        if "entry" in raw and isinstance(raw["entry"], list) and raw["entry"]:
            changes = raw["entry"][0].get("changes", [])
            if changes:
                value = changes[0].get("value", {})
                messages = value.get("messages", [])
                if messages:
                    entry = messages[0]
                contacts = value.get("contacts", [])
                if contacts:
                    entry["sender_name"] = contacts[0].get("profile", {}).get("name", "")

        msg_type = entry.get("type", "text")
        content = ""

        if msg_type == "text":
            content = entry.get("text", {}).get("body", "")
        elif msg_type == "image":
            content = f"[image] {entry.get('image', {}).get('caption', '')}"
        elif msg_type == "document":
            content = f"[document] {entry.get('document', {}).get('filename', '')}"
        elif msg_type == "audio":
            content = "[audio message]"
        elif msg_type == "location":
            loc = entry.get("location", {})
            content = f"[location] {loc.get('latitude')}, {loc.get('longitude')}"
        elif msg_type == "interactive":
            interactive = entry.get("interactive", {})
            if "button_reply" in interactive:
                content = interactive["button_reply"].get("title", "")
            elif "list_reply" in interactive:
                content = interactive["list_reply"].get("title", "")
            else:
                content = "[interactive message]"

        return UnifiedMessage(
            id=entry.get("id", ""),
            role=MessageRole.USER,
            content=[ContentBlock.text_block(content)],
            platform="whatsapp",
            chat_id=entry.get("from", ""),
            sender_id=entry.get("from", ""),
            sender_name=entry.get("sender_name", ""),
        )
