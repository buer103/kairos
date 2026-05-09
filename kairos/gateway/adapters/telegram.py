"""Telegram Bot API adapter — full REST API, webhook + getUpdates polling.

Docs: https://core.telegram.org/bots/api

Features:
  - Send text, photos, documents, media groups
  - Long-polling via getUpdates
  - Webhook setup + verification
  - Message parsing (text, photo, document, location, callback_query)
  - Parse mode (MarkdownV2/HTML)
  - Inline keyboard support
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Any

from kairos.gateway.protocol import (
    UnifiedMessage,
    UnifiedResponse,
    ContentBlock,
    ContentType,
    MessageRole,
    ConnectionState,
)
from kairos.gateway.adapters.base import PlatformAdapter


class TelegramAdapter(PlatformAdapter):
    """Telegram Bot API adapter — zero-dependency REST implementation."""

    platform_name = "telegram"
    API_BASE = "https://api.telegram.org"

    def __init__(self, bot_token: str = "", webhook_url: str = ""):
        super().__init__()
        self._token = bot_token
        self._webhook_url = webhook_url
        self._last_update_id = 0
        self._bot_info: dict[str, Any] = {}

    # ── Connection lifecycle ──────────────────────────────────────────────

    async def connect(self) -> bool:
        """Verify token and fetch bot info via getMe."""
        if not self._token:
            self._state = ConnectionState.ERROR
            return False
        try:
            info = self._api_call("getMe")
            if not info.get("ok"):
                self._state = ConnectionState.ERROR
                return False
            self._bot_info = info.get("result", {})
            self._state = ConnectionState.CONNECTED
            return True
        except Exception:
            self._state = ConnectionState.ERROR
            return False

    async def disconnect(self) -> None:
        """Delete webhook and disconnect."""
        if self._state == ConnectionState.CONNECTED and self._webhook_url:
            try:
                self._api_call("deleteWebhook", {"drop_pending_updates": True})
            except Exception:
                pass
        self._state = ConnectionState.DISCONNECTED

    # ── Message I/O ───────────────────────────────────────────────────────

    async def send(self, chat_id: str, response: UnifiedResponse) -> bool:
        """Send text + optional media to a Telegram chat.

        Auto-splits long messages (>4096 chars) into multiple sendMessage calls.
        Media is sent as photo/document depending on ContentType.
        """
        if not self._token:
            return False

        success = True
        text_chunks = _chunk_text(response.text, 4000)

        for chunk in text_chunks:
            try:
                payload: dict[str, Any] = {
                    "chat_id": chat_id,
                    "text": chunk,
                    "parse_mode": "MarkdownV2" if self._has_markdown(chunk) else "",
                    "disable_web_page_preview": True,
                }
                if not payload["parse_mode"]:
                    del payload["parse_mode"]
                self._api_call("sendMessage", payload)
            except Exception:
                success = False

        # Send media attachments
        for media in response.media:
            try:
                media_type = media.get("type", "photo")
                media_url = media.get("url") or media.get("path", "")
                if not media_url:
                    continue

                if media_type in ("photo", "image"):
                    self._api_call("sendPhoto", {
                        "chat_id": chat_id,
                        "photo": media_url,
                        "caption": media.get("caption", ""),
                    })
                elif media_type in ("document", "file"):
                    self._api_call("sendDocument", {
                        "chat_id": chat_id,
                        "document": media_url,
                        "caption": media.get("caption", ""),
                    })
            except Exception:
                success = False

        return success

    async def receive(self) -> UnifiedMessage | None:
        """Long-poll getUpdates for one message. Returns None if no new messages."""
        if not self._token:
            return None

        try:
            params = {"offset": self._last_update_id + 1, "timeout": 30, "limit": 1}
            result = self._api_call("getUpdates", params)
            if not result.get("ok"):
                return None

            updates = result.get("result", [])
            if not updates:
                return None

            update = updates[0]
            self._last_update_id = update.get("update_id", 0)
            return self.translate_incoming(update)

        except Exception:
            return None

    # ── Webhook ───────────────────────────────────────────────────────────

    def setup_webhook(self, url: str, secret_token: str = "") -> dict:
        """Register a webhook URL with Telegram. Returns API response."""
        params: dict[str, Any] = {"url": url}
        if secret_token:
            params["secret_token"] = secret_token
        return self._api_call("setWebhook", params)

    def verify_webhook(self, headers: dict[str, str], body: bytes) -> bool:
        """Verify webhook request by checking X-Telegram-Bot-Api-Secret-Token."""
        token = headers.get("x-telegram-bot-api-secret-token", "")
        return bool(token)  # Simple token presence check

    # ── Translation ───────────────────────────────────────────────────────

    def translate_incoming(self, raw: dict[str, Any]) -> UnifiedMessage:
        """Parse a Telegram update into UnifiedMessage.

        Handles: message (text/photo/document/location/contact),
                 callback_query, inline_query, edited_message.
        """
        # Callback query (button press)
        if "callback_query" in raw:
            cq = raw["callback_query"]
            msg = cq.get("message", {})
            return UnifiedMessage(
                id=str(cq.get("id", "")),
                role=MessageRole.USER,
                content=[ContentBlock(type=ContentType.TEXT, text=cq.get("data", ""))],
                platform="telegram",
                chat_id=str(msg.get("chat", {}).get("id", "")),
                sender_id=str(cq.get("from", {}).get("id", "")),
                sender_name=_sender_display(cq.get("from", {})),
                raw_payload=cq,
            )

        # Inline query
        if "inline_query" in raw:
            iq = raw["inline_query"]
            return UnifiedMessage(
                id=str(iq.get("id", "")),
                role=MessageRole.USER,
                content=[ContentBlock(type=ContentType.TEXT, text=iq.get("query", ""))],
                platform="telegram",
                chat_id=str(iq.get("from", {}).get("id", "")),
                sender_id=str(iq.get("from", {}).get("id", "")),
                sender_name=_sender_display(iq.get("from", {})),
                raw_payload=iq,
            )

        # Regular message (or edited_message)
        msg = raw.get("message") or raw.get("edited_message") or raw
        msg_id = str(msg.get("message_id", ""))
        chat = msg.get("chat", {})
        chat_id = str(chat.get("id", ""))
        sender = msg.get("from", {})
        sender_id = str(sender.get("id", ""))
        sender_name = _sender_display(sender)

        # Build content blocks
        content: list[ContentBlock] = []

        # Text
        text = msg.get("text") or msg.get("caption", "") or ""
        if text:
            content.append(ContentBlock(type=ContentType.TEXT, text=text))

        # Photo — use largest size
        photos = msg.get("photo", [])
        if photos:
            best = max(photos, key=lambda p: p.get("width", 0) * p.get("height", 0))
            file_id = best.get("file_id", "")
            content.append(ContentBlock(
                type=ContentType.IMAGE,
                url=file_id,
                mime_type="image/jpeg",
            ))

        # Document
        doc = msg.get("document", {})
        if doc:
            content.append(ContentBlock(
                type=ContentType.FILE,
                url=doc.get("file_id", ""),
                text=doc.get("file_name", ""),
                mime_type=doc.get("mime_type", ""),
            ))

        # Audio / Voice
        for key, ct in [("audio", ContentType.AUDIO), ("voice", ContentType.AUDIO)]:
            item = msg.get(key, {})
            if item:
                content.append(ContentBlock(
                    type=ct,
                    url=item.get("file_id", ""),
                    mime_type=item.get("mime_type", ""),
                ))

        # Video / Video note
        for key in ("video", "video_note"):
            item = msg.get(key, {})
            if item:
                content.append(ContentBlock(
                    type=ContentType.VIDEO,
                    url=item.get("file_id", ""),
                    mime_type=item.get("mime_type", ""),
                ))

        # Location
        loc = msg.get("location", {})
        if loc:
            content.append(ContentBlock(
                type=ContentType.TEXT,
                text=f"📍 {loc.get('latitude')}, {loc.get('longitude')}",
            ))

        # Contact
        contact = msg.get("contact", {})
        if contact:
            content.append(ContentBlock(
                type=ContentType.TEXT,
                text=f"👤 {contact.get('first_name', '')} {contact.get('phone_number', '')}",
            ))

        if not content:
            content = [ContentBlock(type=ContentType.TEXT, text="")]

        return UnifiedMessage(
            id=msg_id,
            role=MessageRole.USER,
            content=content,
            platform="telegram",
            chat_id=chat_id,
            thread_id=str(chat.get("id", "")),
            sender_id=sender_id,
            sender_name=sender_name,
            raw_payload=msg,
        )

    def translate_outgoing(self, response: UnifiedResponse) -> dict[str, Any]:
        """Convert UnifiedResponse to Telegram sendMessage-compatible dict."""
        return {
            "text": response.text,
            "parse_mode": "MarkdownV2" if self._has_markdown(response.text) else "",
            "media": [m for m in response.media],
            "confidence": response.confidence,
        }

    # ── Helpers ───────────────────────────────────────────────────────────

    def _api_call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Call a Telegram Bot API method and return parsed JSON."""
        url = f"{self.API_BASE}/bot{self._token}/{method}"
        data = None
        headers = {"Content-Type": "application/json"}

        if params:
            data = json.dumps(params).encode("utf-8")

        req = urllib.request.Request(url, data=data, headers=headers)
        resp = urllib.request.urlopen(req, timeout=15)
        body = resp.read().decode("utf-8")
        return json.loads(body)

    @staticmethod
    def _has_markdown(text: str) -> bool:
        """Check if text contains MarkdownV2 formatting characters."""
        markers = ("*", "_", "`", "[", "~", "||")
        return any(m in text for m in markers)


# ── Helpers ───────────────────────────────────────────────────────────────


def _chunk_text(text: str, max_len: int = 4000) -> list[str]:
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


def _sender_display(sender: dict[str, Any]) -> str:
    """Build display name from Telegram user object."""
    first = sender.get("first_name", "")
    last = sender.get("last_name", "")
    username = sender.get("username", "")
    if first and last:
        name = f"{first} {last}"
    elif first:
        name = first
    elif username:
        name = f"@{username}"
    else:
        name = str(sender.get("id", "unknown"))
    if username:
        name += f" (@{username})"
    return name
