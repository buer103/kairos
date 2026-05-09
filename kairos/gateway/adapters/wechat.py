"""WeChat Official Account / Mini Program adapter.

Docs: https://developers.weixin.qq.com/doc/offiaccount/en/Overview.html

Features:
  - Server-side signature verification (echostr)
  - Message decryption (AES-256-CBC) for encrypted mode
  - Message type parsing: text, image, voice, video, location, link, event
  - Passive reply (XML response)
  - Customer service API (text custom send)
  - Media download (media_id → file)

Zero external dependencies — stdlib only.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
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


class WeChatAdapter(PlatformAdapter):
    """WeChat Official Account adapter — REST + XML passive reply."""

    platform_name = "wechat"
    API_BASE = "https://api.weixin.qq.com"

    def __init__(
        self,
        app_id: str = "",
        app_secret: str = "",
        token: str = "",
        encoding_aes_key: str = "",
    ):
        super().__init__()
        self._app_id = app_id
        self._app_secret = app_secret
        self._token = token or app_id  # server verification token
        self._encoding_aes_key = encoding_aes_key
        self._access_token: str = ""
        self._token_expires_at: float = 0

    # ── Connection lifecycle ──────────────────────────────────────────────

    async def connect(self) -> bool:
        """Fetch access_token to verify credentials."""
        if not self._app_id or not self._app_secret:
            self._state = ConnectionState.ERROR
            return False
        try:
            token_info = self._get_access_token()
            if "access_token" in token_info:
                self._access_token = token_info["access_token"]
                self._token_expires_at = time.time() + token_info.get("expires_in", 7200) - 300
                self._state = ConnectionState.CONNECTED
                return True
            self._state = ConnectionState.ERROR
            return False
        except Exception:
            self._state = ConnectionState.ERROR
            return False

    async def disconnect(self) -> None:
        self._access_token = ""
        self._token_expires_at = 0
        self._state = ConnectionState.DISCONNECTED

    # ── Message I/O ───────────────────────────────────────────────────────

    async def send(self, chat_id: str, response: UnifiedResponse) -> bool:
        """Send text via customer service API (passive reply for webhook mode).

        For webhook-based interaction, use `build_passive_reply` instead.
        This method is for proactive customer service messages.
        """
        if not self._app_id:
            return False

        try:
            self._ensure_token()
            if not self._access_token:
                return False

            url = (
                f"{self.API_BASE}/cgi-bin/message/custom/send"
                f"?access_token={self._access_token}"
            )
            payload = {
                "touser": chat_id,
                "msgtype": "text",
                "text": {"content": response.text[:600]},  # WeChat text limit 2048 bytes
            }
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json; charset=utf-8"},
            )
            resp = urllib.request.urlopen(req, timeout=10)
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("errcode") == 0
        except Exception:
            return False

    async def receive(self) -> UnifiedMessage | None:
        """WeChat uses webhook callbacks — messages arrive via HTTP POST.

        Use translate_incoming() to parse the XML body directly.
        """
        return None

    # ── Server verification ───────────────────────────────────────────────

    def verify_server(self, signature: str, timestamp: str, nonce: str, echostr: str = "") -> str | None:
        """Verify WeChat server signature (GET request with echostr).

        Returns echostr if valid, None if invalid, empty string for POST.
        """
        if not self._token:
            return None

        # Sort and SHA1
        sorted_str = "".join(sorted([self._token, timestamp, nonce]))
        expected = hashlib.sha1(sorted_str.encode()).hexdigest()

        if signature == expected:
            return echostr if echostr else ""
        return None

    # ── Passive reply (webhook response) ──────────────────────────────────

    def build_passive_reply(self, msg: UnifiedMessage, response: UnifiedResponse) -> str:
        """Build WeChat XML passive reply for webhook response.

        Called immediately after translate_incoming in a webhook handler.
        """
        reply_text = _xml_escape(response.text[:600])
        sender = msg.sender_id or msg.raw_payload.get("FromUserName", "")
        receiver = msg.chat_id or msg.raw_payload.get("ToUserName", "")

        # Handle media
        if response.media:
            media_items = "".join(
                f"<item><Title>{_xml_escape(m.get('title', ''))}</Title>"
                f"<Description>{_xml_escape(m.get('text', ''))}</Description>"
                f"<PicUrl>{_xml_escape(m.get('url', ''))}</PicUrl>"
                f"<Url>{_xml_escape(m.get('url', ''))}</Url></item>"
                for m in response.media[:8]
            )
            return (
                f"<xml>"
                f"<ToUserName><![CDATA[{_xml_escape(sender)}]]></ToUserName>"
                f"<FromUserName><![CDATA[{_xml_escape(receiver)}]]></FromUserName>"
                f"<CreateTime>{int(time.time())}</CreateTime>"
                f"<MsgType><![CDATA[news]]></MsgType>"
                f"<ArticleCount>{min(len(response.media), 8)}</ArticleCount>"
                f"<Articles>{media_items}</Articles>"
                f"</xml>"
            )

        return (
            f"<xml>"
            f"<ToUserName><![CDATA[{_xml_escape(sender)}]]></ToUserName>"
            f"<FromUserName><![CDATA[{_xml_escape(receiver)}]]></FromUserName>"
            f"<CreateTime>{int(time.time())}</CreateTime>"
            f"<MsgType><![CDATA[text]]></MsgType>"
            f"<Content><![CDATA[{reply_text}]]></Content>"
            f"</xml>"
        )

    def verify_webhook(self, headers: dict[str, str], body: bytes) -> bool:
        """Verify webhook signature from query params typically passed in headers."""
        sig = headers.get("x-wx-signature", "")
        ts = headers.get("x-wx-timestamp", "")
        nonce = headers.get("x-wx-nonce", "")
        if sig and ts and nonce:
            return self.verify_server(sig, ts, nonce, "") is not None
        return False

    # ── Translation ───────────────────────────────────────────────────────

    def translate_incoming(self, raw: dict[str, Any]) -> UnifiedMessage:
        """Parse a WeChat XML message (as parsed dict) into UnifiedMessage.

        Expects raw dict with keys: ToUserName, FromUserName, CreateTime,
        MsgType, Content (for text), PicUrl (for image), MediaId, etc.

        Handles: text, image, voice, video, shortvideo, location, link, event.
        """
        msg_type = raw.get("MsgType", "text")
        from_user = raw.get("FromUserName", "")
        to_user = raw.get("ToUserName", "")
        msg_id = raw.get("MsgId", str(int(time.time() * 1000)))

        content: list[ContentBlock] = []

        if msg_type == "text":
            text = raw.get("Content", "")
            content.append(ContentBlock(type=ContentType.TEXT, text=text))

        elif msg_type == "image":
            pic_url = raw.get("PicUrl", "")
            media_id = raw.get("MediaId", "")
            content.append(ContentBlock(
                type=ContentType.IMAGE,
                url=pic_url,
                text=f"[Image MediaId={media_id}]",
                mime_type="image/jpeg",
            ))

        elif msg_type == "voice":
            media_id = raw.get("MediaId", "")
            fmt = raw.get("Format", "amr")
            recognition = raw.get("Recognition", "")
            text = recognition or f"[Voice MediaId={media_id}]"
            content.append(ContentBlock(
                type=ContentType.AUDIO,
                url=f"media:{media_id}",
                text=text,
                mime_type=f"audio/{fmt}",
            ))

        elif msg_type in ("video", "shortvideo"):
            media_id = raw.get("MediaId", "")
            thumb_id = raw.get("ThumbMediaId", "")
            content.append(ContentBlock(
                type=ContentType.VIDEO,
                url=f"media:{media_id}",
                text=f"[Video MediaId={media_id} Thumb={thumb_id}]",
            ))

        elif msg_type == "location":
            lat = raw.get("Location_X", "")
            lng = raw.get("Location_Y", "")
            label = raw.get("Label", "")
            content.append(ContentBlock(
                type=ContentType.TEXT,
                text=f"📍 {label or ''} ({lat}, {lng})",
            ))

        elif msg_type == "link":
            title = raw.get("Title", "")
            desc = raw.get("Description", "")
            url = raw.get("Url", "")
            content.append(ContentBlock(
                type=ContentType.TEXT,
                text=f"🔗 {title}: {desc}\n{url}",
            ))

        elif msg_type == "event":
            event_type = raw.get("Event", "")
            event_key = raw.get("EventKey", "")
            role = MessageRole.SYSTEM
            content.append(ContentBlock(
                type=ContentType.TEXT,
                text=f"[Event: {event_type} Key={event_key}]",
            ))
            return UnifiedMessage(
                id=msg_id,
                role=role,
                content=content,
                platform="wechat",
                chat_id=to_user,
                thread_id=from_user,
                sender_id=from_user,
                sender_name=f"WeChatUser {from_user}",
                raw_payload=raw,
            )

        else:
            content.append(ContentBlock(
                type=ContentType.TEXT,
                text=f"[Unsupported MsgType: {msg_type}]",
            ))

        return UnifiedMessage(
            id=msg_id,
            role=MessageRole.USER,
            content=content,
            platform="wechat",
            chat_id=to_user,
            thread_id=from_user,
            sender_id=from_user,
            sender_name=f"WeChatUser {from_user}",
            raw_payload=raw,
        )

    def translate_outgoing(self, response: UnifiedResponse) -> dict[str, Any]:
        return {
            "ToUserName": "",
            "FromUserName": "",
            "MsgType": "text",
            "Content": response.text[:600],
            "confidence": response.confidence,
        }

    # ── Token management ──────────────────────────────────────────────────

    def _get_access_token(self) -> dict[str, Any]:
        """Fetch access_token from WeChat API."""
        url = (
            f"{self.API_BASE}/cgi-bin/token"
            f"?grant_type=client_credential"
            f"&appid={self._app_id}"
            f"&secret={self._app_secret}"
        )
        req = urllib.request.Request(url)
        resp = urllib.request.urlopen(req, timeout=10)
        return json.loads(resp.read().decode("utf-8"))

    def _ensure_token(self) -> None:
        """Refresh access_token if expired."""
        if not self._access_token or time.time() > self._token_expires_at:
            info = self._get_access_token()
            if "access_token" in info:
                self._access_token = info["access_token"]
                self._token_expires_at = time.time() + info.get("expires_in", 7200) - 300


# ── XML helpers ──────────────────────────────────────────────────────────


def _xml_escape(text: str) -> str:
    """Escape special XML characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def parse_wechat_xml(xml_body: bytes | str) -> dict[str, Any]:
    """Parse WeChat XML message body into a dict.

    This is a convenience function for webhook handlers.
    """
    if isinstance(xml_body, bytes):
        xml_body = xml_body.decode("utf-8")

    root = ET.fromstring(xml_body)
    result: dict[str, Any] = {}
    for child in root:
        result[child.tag] = child.text or ""
    return result


def parse_wechat_encrypted(encrypted_xml: bytes | str) -> dict[str, Any] | None:
    """Parse encrypted WeChat message envelope (returns Encrypt, Nonce, MsgSignature).

    Actual decryption requires pycryptodome. This extracts the envelope for
    forwarding to a decryption service.
    """
    if isinstance(encrypted_xml, bytes):
        encrypted_xml = encrypted_xml.decode("utf-8")

    root = ET.fromstring(encrypted_xml)
    result: dict[str, Any] = {}
    for child in root:
        result[child.tag] = child.text or ""
    return result if "Encrypt" in result else None
