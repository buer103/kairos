"""Tests for Telegram, WeChat, and Slack gateway adapters.

Tests translate_incoming parsing, connect/send flows, and edge cases.
Written for the actual adapter implementations (not stubs).
"""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import patch, MagicMock, call

import pytest

from kairos.gateway.adapters import TelegramAdapter, WeChatAdapter, SlackAdapter
from kairos.gateway.adapters.base import CLIAdapter
from kairos.gateway.protocol import (
    UnifiedMessage,
    UnifiedResponse,
    ContentBlock,
    ContentType,
    MessageRole,
    ConnectionState,
)


def make_response(text: str = "hello") -> UnifiedResponse:
    return UnifiedResponse(text=text, confidence=0.95)


def mock_urlopen(body: dict | str = None, status: int = 200):
    """Helper to mock urllib.request.urlopen with JSON or string body."""
    if isinstance(body, dict):
        body_bytes = json.dumps(body).encode()
    elif isinstance(body, str):
        body_bytes = body.encode()
    else:
        body_bytes = b'{"ok":true}'

    return patch(
        "urllib.request.urlopen",
        return_value=MagicMock(
            __enter__=MagicMock(return_value=MagicMock(
                status=status,
                read=MagicMock(return_value=body_bytes),
                headers={},
            )),
            __exit__=MagicMock(return_value=None),
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════
# TelegramAdapter
# ═══════════════════════════════════════════════════════════════════════════

class TestTelegramTranslateIncoming:
    """Telegram Update -> UnifiedMessage translation."""

    def test_text_message(self):
        """Parse a basic Telegram text message Update."""
        adapter = TelegramAdapter(bot_token="test")
        update = {
            "message": {
                "message_id": 123,
                "from": {"id": 456, "first_name": "Alice", "username": "alice_user"},
                "chat": {"id": 789, "type": "private"},
                "date": 1700000000,
                "text": "Hello Kairos!",
            }
        }
        msg = adapter.translate_incoming(update)

        assert msg.role == MessageRole.USER
        assert msg.platform == "telegram"
        assert msg.id == "123"
        assert "Hello Kairos" in msg.text
        assert msg.sender_name == "Alice (@alice_user)"
        assert msg.sender_id == "456"
        assert msg.chat_id == "789"

    def test_photo_message(self):
        """Parse a Telegram photo message."""
        adapter = TelegramAdapter(bot_token="test")
        update = {
            "message": {
                "message_id": 124,
                "from": {"id": 100, "first_name": "Bob"},
                "chat": {"id": 200, "type": "private"},
                "photo": [
                    {"file_id": "small_id", "width": 100, "height": 100},
                    {"file_id": "large_id", "width": 800, "height": 600},
                ],
                "caption": "Check this out!",
            }
        }
        msg = adapter.translate_incoming(update)

        assert msg.text == "Check this out!"
        assert any(b.type == ContentType.IMAGE for b in msg.content)
        assert any("large_id" in b.url for b in msg.content if b.type == ContentType.IMAGE)
        assert msg.sender_name == "Bob"

    def test_callback_query(self):
        """Parse a Telegram callback query (inline button press)."""
        adapter = TelegramAdapter(bot_token="test")
        update = {
            "callback_query": {
                "id": "cb_001",
                "from": {"id": 999, "username": "clicker"},
                "message": {
                    "message_id": 500,
                    "chat": {"id": 200, "type": "private"},
                },
                "data": "menu_settings",
            }
        }
        msg = adapter.translate_incoming(update)

        assert msg.role == MessageRole.USER
        assert msg.platform == "telegram"
        assert msg.text == "menu_settings"
        assert msg.sender_id == "999"
        assert msg.chat_id == "200"

    def test_inline_query(self):
        """Parse a Telegram inline query."""
        adapter = TelegramAdapter(bot_token="test")
        update = {
            "inline_query": {
                "id": "iq_001",
                "from": {"id": 111, "first_name": "Charlie"},
                "query": "search term",
                "offset": "",
            }
        }
        msg = adapter.translate_incoming(update)

        assert msg.text == "search term"
        assert msg.sender_id == "111"

    def test_no_content_message(self):
        """Parse a message with no text/media (e.g., sticker)."""
        adapter = TelegramAdapter(bot_token="test")
        update = {
            "message": {
                "message_id": 125,
                "from": {"id": 300},
                "chat": {"id": 400},
                "sticker": {"file_id": "sticker1", "emoji": "👍"},
            }
        }
        msg = adapter.translate_incoming(update)
        # Should not crash, should have an empty text block
        assert msg.role == MessageRole.USER
        assert msg.platform == "telegram"


class TestTelegramAdapterLifecycle:
    """Telegram adapter connect/disconnect/state."""

    def test_no_token_error(self):
        adapter = TelegramAdapter()
        assert adapter.state == ConnectionState.DISCONNECTED

    def test_platform_name(self):
        adapter = TelegramAdapter()
        assert adapter.platform_name == "telegram"

    def test_receive_returns_none_when_no_token(self):
        """receive() returns None when polling not available."""
        adapter = TelegramAdapter()
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(adapter.receive())
        assert result is None  # no token = no polling


# ═══════════════════════════════════════════════════════════════════════════
# WeChatAdapter
# ═══════════════════════════════════════════════════════════════════════════

class TestWeChatTranslateIncoming:
    """WeChat XML-like dict -> UnifiedMessage translation."""

    def test_text_message(self):
        """Parse a WeChat text message."""
        adapter = WeChatAdapter(app_id="wx123", app_secret="secret")
        payload = {
            "ToUserName": "gh_test",
            "FromUserName": "oXyz_user",
            "CreateTime": "1700000000",
            "MsgType": "text",
            "Content": "Hello WeChat!",
            "MsgId": "msg_001",
        }
        msg = adapter.translate_incoming(payload)

        assert msg.role == MessageRole.USER
        assert msg.platform == "wechat"
        assert msg.text == "Hello WeChat!"
        assert msg.sender_id == "oXyz_user"
        assert msg.chat_id == "gh_test"

    def test_image_message(self):
        """Parse a WeChat image message."""
        adapter = WeChatAdapter()
        payload = {
            "ToUserName": "gh_test",
            "FromUserName": "oXyz_user",
            "MsgType": "image",
            "PicUrl": "https://example.com/img.jpg",
            "MediaId": "media_abc",
            "MsgId": "msg_002",
        }
        msg = adapter.translate_incoming(payload)

        assert any(b.type == ContentType.IMAGE for b in msg.content)
        img_block = next(b for b in msg.content if b.type == ContentType.IMAGE)
        assert "example.com/img.jpg" in img_block.url

    def test_subscribe_event(self):
        """Parse a WeChat subscribe (follow) event."""
        adapter = WeChatAdapter()
        payload = {
            "ToUserName": "gh_test",
            "FromUserName": "oXyz_user",
            "MsgType": "event",
            "Event": "subscribe",
            "EventKey": "",
        }
        msg = adapter.translate_incoming(payload)

        assert msg.role == MessageRole.SYSTEM
        assert "subscribe" in msg.text

    def test_location_message(self):
        """Parse a WeChat location message."""
        adapter = WeChatAdapter()
        payload = {
            "ToUserName": "gh_test",
            "FromUserName": "oUser",
            "MsgType": "location",
            "Location_X": "39.9042",
            "Location_Y": "116.4074",
            "Label": "Beijing",
            "MsgId": "msg_003",
        }
        msg = adapter.translate_incoming(payload)

        assert "39.9042" in msg.text
        assert "Beijing" in msg.text


class TestWeChatConnect:
    """WeChat connect/disconnect behavior."""

    def test_no_app_id_error(self):
        adapter = WeChatAdapter()
        assert adapter.state == ConnectionState.DISCONNECTED

    def test_platform_name(self):
        adapter = WeChatAdapter()
        assert adapter.platform_name == "wechat"


class TestWeChatServerVerification:
    """WeChat server signature verification."""

    def test_valid_signature(self):
        """Verify a valid WeChat server signature."""
        adapter = WeChatAdapter(token="my_token")
        # SHA1 of sorted("my_token" + "1234567890" + "random_nonce")
        import hashlib
        sorted_str = "".join(sorted(["my_token", "1234567890", "random_nonce"]))
        expected = hashlib.sha1(sorted_str.encode()).hexdigest()

        result = adapter.verify_server(expected, "1234567890", "random_nonce", "echo_ok")
        assert result == "echo_ok"

    def test_invalid_signature(self):
        """Reject invalid signature."""
        adapter = WeChatAdapter(token="my_token")
        result = adapter.verify_server("wrong", "1234567890", "random_nonce", "echo")
        assert result is None

    def test_no_token(self):
        """No token configured — verification fails."""
        adapter = WeChatAdapter()  # no token
        result = adapter.verify_server("any", "ts", "nonce", "echo")
        assert result is None


class TestWeChatPassiveReply:
    """WeChat XML passive reply generation."""

    def test_text_reply(self):
        """Build a text passive reply."""
        adapter = WeChatAdapter(app_id="wx123")
        msg = UnifiedMessage(
            id="orig_001",
            role=MessageRole.USER,
            content=[ContentBlock.text_block("incoming")],
            platform="wechat",
            chat_id="app_id",
            sender_id="user_id",
            raw_payload={"FromUserName": "user_id", "ToUserName": "app_id"},
        )
        response = UnifiedResponse(text="Reply text")
        xml = adapter.build_passive_reply(msg, response)

        assert "<xml>" in xml
        assert "<![CDATA[Reply text]]>" in xml
        assert "<![CDATA[user_id]]>" in xml  # ToUserName = sender
        assert "<![CDATA[app_id]]>" in xml  # FromUserName = receiver

    def test_media_reply(self):
        """Build a news article reply for media."""
        adapter = WeChatAdapter()
        msg = UnifiedMessage(
            id="orig_001",
            role=MessageRole.USER,
            content=[ContentBlock.text_block("incoming")],
            platform="wechat",
            sender_id="user_id",
            chat_id="app_id",
            raw_payload={"FromUserName": "user_id", "ToUserName": "app_id"},
        )
        response = UnifiedResponse(
            text="Articles",
            media=[{"url": "https://example.com/1.jpg", "title": "Title", "text": "Desc"}],
        )
        xml = adapter.build_passive_reply(msg, response)

        assert "<MsgType><![CDATA[news]]></MsgType>" in xml
        assert "1.jpg" in xml


# ═══════════════════════════════════════════════════════════════════════════
# SlackAdapter
# ═══════════════════════════════════════════════════════════════════════════

class TestSlackTranslateIncoming:
    """Slack Events API -> UnifiedMessage translation."""

    def test_message_event(self):
        """Parse a Slack message event."""
        adapter = SlackAdapter(bot_token="xoxb-test")
        payload = {
            "token": "verification_token",
            "team_id": "T123",
            "event": {
                "type": "message",
                "user": "U456",
                "text": "Hello from Slack!",
                "channel": "C789",
                "ts": "1700000000.000001",
            },
        }
        msg = adapter.translate_incoming(payload)

        assert msg.role == MessageRole.USER
        assert msg.platform == "slack"
        assert "Hello from Slack" in msg.text
        assert msg.sender_id == "U456"
        assert msg.chat_id == "C789"

    def test_url_verification_event(self):
        """Parse a Slack URL verification challenge."""
        adapter = SlackAdapter(bot_token="xoxb-test")
        payload = {
            "token": "tok",
            "challenge": "challenge_code_123",
            "type": "url_verification",
        }
        msg = adapter.translate_incoming(payload)

        assert msg.role == MessageRole.SYSTEM
        assert msg.raw_payload["type"] == "url_verification"

    def test_app_mention_event(self):
        """Parse a Slack app_mention event."""
        adapter = SlackAdapter(bot_token="xoxb-test")
        payload = {
            "event": {
                "type": "app_mention",
                "user": "U789",
                "text": "<@BOT> help me",
                "channel": "C001",
            },
        }
        msg = adapter.translate_incoming(payload)

        # Slack adapter cleans up @bot mentions, but keeps "help me"
        assert "help me" in msg.text
        assert msg.sender_id == "U789"

    def test_file_share_event(self):
        """Parse a Slack file_share message."""
        adapter = SlackAdapter(bot_token="xoxb-test")
        payload = {
            "event": {
                "type": "message",
                "subtype": "file_share",
                "user": "U111",
                "text": "Shared a file",
                "files": [{"name": "doc.pdf", "url_private": "https://slack.com/file"}],
                "channel": "C002",
            },
        }
        msg = adapter.translate_incoming(payload)

        assert "Shared a file" in msg.text
        assert msg.sender_id == "U111"


class TestSlackAdapterLifecycle:
    """Slack adapter basic lifecycle."""

    def test_no_token(self):
        adapter = SlackAdapter()
        assert adapter.state == ConnectionState.DISCONNECTED

    def test_platform_name(self):
        adapter = SlackAdapter()
        assert adapter.platform_name == "slack"

    def test_receive_returns_none(self):
        adapter = SlackAdapter(bot_token="xoxb-test")
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(adapter.receive())
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# CLIAdapter (still in base.py)
# ═══════════════════════════════════════════════════════════════════════════

class TestCLIAdapter:
    """CLI adapter tests."""

    def test_always_connected(self):
        adapter = CLIAdapter()
        assert adapter.state == ConnectionState.CONNECTED
        assert adapter.platform_name == "cli"

    def test_send_prints(self, capsys):
        adapter = CLIAdapter()
        response = make_response("CLI test response")
        import asyncio
        asyncio.get_event_loop().run_until_complete(adapter.send("", response))
        captured = capsys.readouterr()
        assert "CLI test response" in captured.out

    def test_translate_incoming(self):
        adapter = CLIAdapter()
        raw = {
            "id": "manual",
            "role": "user",
            "content": [ContentBlock.text_block("manual input")],
            "platform": "cli",
            "chat_id": "test",
            "thread_id": "t1",
            "sender_id": "s1",
            "sender_name": "Test",
        }
        msg = adapter.translate_incoming(raw)
        assert msg.platform == "cli"
        assert msg.chat_id == "test"


# ═══════════════════════════════════════════════════════════════════════════
# Response translation
# ═══════════════════════════════════════════════════════════════════════════

class TestTranslateOutgoing:
    """UnifiedResponse -> platform-specific format."""

    def test_telegram_outgoing(self):
        adapter = TelegramAdapter()
        response = UnifiedResponse(text="Hello World", confidence=0.9)
        result = adapter.translate_outgoing(response)
        assert result["text"] == "Hello World"
        assert result["confidence"] == 0.9

    def test_wechat_outgoing(self):
        adapter = WeChatAdapter()
        response = UnifiedResponse(text="Hello", confidence=0.8)
        result = adapter.translate_outgoing(response)
        assert result["Content"] == "Hello"
        assert result["MsgType"] == "text"

    def test_slack_outgoing(self):
        adapter = SlackAdapter()
        response = UnifiedResponse(text="Hi there", confidence=0.7)
        result = adapter.translate_outgoing(response)
        assert "Hi there" in str(result)
