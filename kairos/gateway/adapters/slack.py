"""
Slack Web API adapter.

Uses Slack Web API via plain HTTP (stdlib only, no external deps).
Supports Events API webhook verification, chat.postMessage, and
interactive payloads.

Docs: https://api.slack.com/
"""

from __future__ import annotations

import hashlib
import hmac
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
    ContentType,
)
from kairos.gateway.adapters.base import PlatformAdapter


class SlackAdapter(PlatformAdapter):
    """Slack Bot adapter — Web API + Events API.

    Requires a Slack Bot Token (xoxb-...) and optionally a Signing Secret
    for verifying Events API requests.

    Two modes:
      - Events API: translate_incoming() on HTTP POST body + verify_request()
      - Active send: send() via chat.postMessage
    """

    platform_name = "slack"
    API_BASE = "https://slack.com/api"

    # Slack limits
    MAX_TEXT_LENGTH = 40000  # chat.postMessage block/text limit
    MAX_BLOCKS = 50

    def __init__(
        self,
        bot_token: str = "",
        signing_secret: str = "",
        app_token: str = "",
    ):
        """
        Args:
            bot_token: Slack Bot User OAuth Token (xoxb-...)
            signing_secret: Slack Signing Secret for request verification
            app_token: Slack App-Level Token (xapp-...) for Socket Mode
        """
        super().__init__()
        self._token = bot_token
        self._signing_secret = signing_secret
        self._app_token = app_token
        self._bot_user_id: str = ""
        self._team_id: str = ""

    # ── Connection ──────────────────────────────────────────────

    async def connect(self) -> bool:
        """Validate bot token by calling auth.test."""
        if not self._token:
            self._state = ConnectionState.ERROR
            return False
        try:
            resp = self._api_call("auth.test")
            if resp.get("ok"):
                self._bot_user_id = resp.get("user_id", "")
                self._team_id = resp.get("team_id", "")
                self._state = ConnectionState.CONNECTED
                return True
            self._state = ConnectionState.ERROR
            return False
        except Exception:
            self._state = ConnectionState.ERROR
            return False

    async def disconnect(self) -> None:
        self._state = ConnectionState.DISCONNECTED

    # ── Send ────────────────────────────────────────────────────

    async def send(self, chat_id: str, response: UnifiedResponse) -> bool:
        """Send a message to a Slack channel or user.

        Uses chat.postMessage.  Supports text with optional blocks
        and media attachments.

        Args:
            chat_id: Channel ID (C...), DM channel (D...), or user ID (U...)
            response: UnifiedResponse to send

        Returns:
            True if sent successfully.
        """
        if not self._token:
            return False

        try:
            payload: dict[str, Any] = {
                "channel": chat_id,
                "text": response.text[:self.MAX_TEXT_LENGTH],
            }

            # Add blocks for rich formatting
            blocks = _build_message_blocks(response)
            if blocks:
                payload["blocks"] = json.dumps(blocks)

            # Attach media as blocks
            if response.media_url:
                media_block = _build_media_block(response.media_url, response.media)
                if media_block:
                    if "blocks" in payload:
                        existing = json.loads(payload["blocks"]) if isinstance(
                            payload["blocks"], str
                        ) else payload["blocks"]
                        existing.append(media_block)
                        payload["blocks"] = json.dumps(existing)
                    else:
                        payload["blocks"] = json.dumps([media_block])

            # If no explicit blocks, create a simple text section
            if "blocks" not in payload:
                payload["blocks"] = json.dumps([
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": response.text[:3000],
                        },
                    }
                ])

            # Send as thread reply if thread_ts is in metadata
            thread_ts = response.metadata.get("thread_ts", "")
            if thread_ts:
                payload["thread_ts"] = thread_ts

            return self._api_call("chat.postMessage", payload).get("ok", False)

        except Exception:
            return False

    async def reply_in_thread(
        self, channel: str, thread_ts: str, text: str
    ) -> bool:
        """Convenience method to reply in a thread."""
        return await self.send(
            channel,
            UnifiedResponse(
                text=text,
                metadata={"thread_ts": thread_ts},
            ),
        )

    # ── Receive ─────────────────────────────────────────────────

    async def receive(self) -> UnifiedMessage | None:
        """Events API: messages arrive via HTTP callback.

        Returns None — use translate_incoming() on the HTTP POST body.
        """
        return None

    # ── Translate ───────────────────────────────────────────────

    def translate_incoming(self, raw: dict[str, Any]) -> UnifiedMessage:
        """Parse a Slack Events API or Interactive payload into UnifiedMessage.

        Handles:
          - URL verification challenge
          - Event callbacks (message, app_mention, etc.)
          - Interactive payloads (block_actions, view_submission, etc.)
          - Slash commands
        """
        # URL verification challenge
        if raw.get("type") == "url_verification":
            return UnifiedMessage(
                id="verify",
                role=MessageRole.SYSTEM,
                content=[ContentBlock.text_block(raw.get("challenge", ""))],
                platform="slack",
                chat_id="",
                raw_payload={"challenge": raw.get("challenge", ""), "type": "url_verification"},
            )

        # Interactive payload (block_actions, view_submission, etc.)
        if raw.get("type") in (
            "block_actions",
            "view_submission",
            "view_closed",
            "message_action",
            "shortcut",
        ):
            return self._translate_interactive(raw)

        # Slash command (comes as form-encoded, but may be parsed to dict)
        if raw.get("command"):
            return self._translate_slash_command(raw)

        # Event callback
        if raw.get("type") == "event_callback":
            return self._translate_event(raw)

        # Direct message event (top-level)
        event = raw.get("event", raw)

        return self._translate_message_event(event)

    def _translate_event(self, raw: dict[str, Any]) -> UnifiedMessage:
        """Parse an Event Callback."""
        event = raw.get("event", {})

        event_type = event.get("type", "")

        # Filter out bot's own messages
        if event.get("subtype") == "bot_message":
            return UnifiedMessage(
                id=event.get("event_ts", ""),
                role=MessageRole.ASSISTANT,
                content=[ContentBlock.text_block(event.get("text", ""))],
                platform="slack",
                chat_id=event.get("channel", ""),
                thread_id=event.get("thread_ts", event.get("ts", "")),
                sender_id=event.get("bot_id", ""),
                sender_name="bot",
            )

        # App mention
        if event_type == "app_mention":
            return self._translate_app_mention(event)

        # Message
        if event_type == "message":
            return self._translate_message_event(event)

        # Member joined channel
        if event_type == "member_joined_channel":
            return UnifiedMessage(
                id=event.get("event_ts", ""),
                role=MessageRole.SYSTEM,
                content=[ContentBlock.text_block(
                    f"[joined] <@{event.get('user', '')}> joined <#{event.get('channel', '')}>"
                )],
                platform="slack",
                chat_id=event.get("channel", ""),
                sender_id=event.get("user", ""),
                sender_name=event.get("user", ""),
            )

        # Unknown event
        return UnifiedMessage(
            id=raw.get("event_id", event.get("event_ts", "")),
            role=MessageRole.SYSTEM,
            content=[ContentBlock.text_block(f"[{event_type}]")],
            platform="slack",
            chat_id=event.get("channel", ""),
            raw_payload=event,
        )

    def _translate_message_event(self, event: dict[str, Any]) -> UnifiedMessage:
        """Parse a message event."""
        text = event.get("text", "")
        subtype = event.get("subtype", "")
        channel = event.get("channel", "")
        user = event.get("user", "")

        # Message changed (edit)
        if subtype == "message_changed":
            changed = event.get("message", {})
            text = changed.get("text", text)
            user = changed.get("user", user)

        # Message deleted
        if subtype == "message_deleted":
            return UnifiedMessage(
                id=event.get("event_ts", event.get("ts", "")),
                role=MessageRole.SYSTEM,
                content=[ContentBlock.text_block("[message deleted]")],
                platform="slack",
                chat_id=event.get("channel", ""),
                thread_id=event.get("thread_ts", ""),
            )

        # File shared
        if subtype == "file_share":
            files = event.get("files", [])
            if files:
                file_info = files[0]
                text = text or f"[file: {file_info.get('title', file_info.get('name', 'file'))}]"

        # Thread broadcast
        if subtype == "thread_broadcast":
            pass  # handled normally

        # Clean <@U...> mentions from text for display
        content_text = _clean_slack_mentions(text)

        content_blocks: list[ContentBlock] = [ContentBlock.text_block(content_text)]

        # Add file blocks
        for f in event.get("files", []):
            if f.get("mimetype", "").startswith("image/"):
                content_blocks.append(ContentBlock(
                    type=ContentType.IMAGE,
                    url=f.get("url_private", f.get("permalink", "")),
                    text=f.get("title", ""),
                ))
            else:
                content_blocks.append(ContentBlock(
                    type=ContentType.FILE,
                    url=f.get("url_private", f.get("permalink", "")),
                    text=f.get("title", f.get("name", "file")),
                ))

        return UnifiedMessage(
            id=event.get("event_ts", event.get("ts", "")),
            role=MessageRole.USER,
            content=content_blocks,
            platform="slack",
            chat_id=channel,
            thread_id=event.get("thread_ts", event.get("ts", "")),
            sender_id=user,
            sender_name="",  # Could be resolved via users.info
            raw_payload=event,
        )

    def _translate_app_mention(self, event: dict[str, Any]) -> UnifiedMessage:
        """Parse an app_mention event."""
        text = event.get("text", "")
        # Strip <@BOT_ID> from the beginning
        if self._bot_user_id:
            text = text.replace(f"<@{self._bot_user_id}>", "").strip()
        else:
            # Remove any <@U...> mention from text
            import re
            text = re.sub(r"<@[A-Z0-9]+>", "", text).strip()

        content_text = _clean_slack_mentions(text)

        return UnifiedMessage(
            id=event.get("event_ts", event.get("ts", "")),
            role=MessageRole.USER,
            content=[ContentBlock.text_block(content_text)],
            platform="slack",
            chat_id=event.get("channel", ""),
            thread_id=event.get("thread_ts", event.get("ts", "")),
            sender_id=event.get("user", ""),
            sender_name="",
            raw_payload=event,
        )

    def _translate_interactive(self, raw: dict[str, Any]) -> UnifiedMessage:
        """Parse an interactive payload (block_actions, view_submission, etc.)."""
        interactive_type = raw.get("type", "interactive")
        user_info = raw.get("user", {})
        user_id = user_info.get("id", "") if isinstance(user_info, dict) else str(user_info)
        user_name = user_info.get("username", "") if isinstance(user_info, dict) else ""

        content_text = ""
        raw_payload: dict[str, Any] = {"type": interactive_type}

        if interactive_type == "block_actions":
            actions = raw.get("actions", [])
            if actions:
                action = actions[0]
                content_text = action.get("value", action.get("text", {}).get("text", ""))
                raw_payload["action_id"] = action.get("action_id", "")
                raw_payload["block_id"] = action.get("block_id", "")

        elif interactive_type == "view_submission":
            view = raw.get("view", {})
            state = view.get("state", {}).get("values", {})
            content_text = json.dumps(state, ensure_ascii=False) if state else "[view_submission]"
            raw_payload["view"] = view.get("id", "")

        elif interactive_type == "message_action":
            content_text = raw.get("callback_id", "")
            raw_payload["callback_id"] = content_text

        else:
            content_text = f"[{interactive_type}]"

        channel = raw.get("channel", {})
        channel_id = channel.get("id", "") if isinstance(channel, dict) else str(channel)

        return UnifiedMessage(
            id=raw.get("trigger_id", raw.get("callback_id", "")),
            role=MessageRole.USER,
            content=[ContentBlock.text_block(content_text)],
            platform="slack",
            chat_id=channel_id,
            sender_id=user_id,
            sender_name=user_name,
            raw_payload=raw_payload,
        )

    def _translate_slash_command(self, raw: dict[str, Any]) -> UnifiedMessage:
        """Parse a Slack slash command."""
        command = raw.get("command", "")
        text = raw.get("text", "")

        return UnifiedMessage(
            id=raw.get("trigger_id", ""),
            role=MessageRole.USER,
            content=[ContentBlock.text_block(f"{command} {text}".strip())],
            platform="slack",
            chat_id=raw.get("channel_id", ""),
            sender_id=raw.get("user_id", ""),
            sender_name=raw.get("user_name", ""),
            raw_payload={"command": command, "text": text, "trigger_id": raw.get("trigger_id", "")},
        )

    # ── Request Verification ────────────────────────────────────

    def verify_request(
        self,
        body: str,
        timestamp: str,
        signature: str,
    ) -> bool:
        """Verify an Events API request using Slack's signing secret.

        Per Slack docs: signature = "v0=" + hex(hmac_sha256(signing_secret, "v0:ts:body"))

        Args:
            body: Raw HTTP request body as string
            timestamp: X-Slack-Request-Timestamp header value
            signature: X-Slack-Signature header value (e.g., "v0=abc123...")

        Returns:
            True if the signature is valid.
        """
        if not self._signing_secret:
            return True  # No verification if secret not configured

        # Reject old timestamps (> 5 minutes)
        try:
            ts = int(timestamp)
            if abs(time.time() - ts) > 300:
                return False
        except (ValueError, TypeError):
            return False

        sig_basestring = f"v0:{timestamp}:{body}"
        computed = "v0=" + hmac.new(
            self._signing_secret.encode("utf-8"),
            sig_basestring.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(computed, signature)

    # ── HTTP helpers ────────────────────────────────────────────

    def _api_call(
        self, method: str, data: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Call a Slack Web API method.

        Args:
            method: API method name (e.g., 'chat.postMessage')
            data: POST body dict

        Returns:
            Parsed JSON response dict.
        """
        url = f"{self.API_BASE}/{method}"

        headers: dict[str, str] = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "Kairos/1.0",
        }

        if data is None:
            data = {}

        # For GET-only methods
        get_methods = {"auth.test", "users.info", "conversations.info", "conversations.list"}
        if method in get_methods and not data:
            req = urllib.request.Request(url, headers=headers)
        else:
            encoded = json.dumps(data, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(url, data=encoded, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            try:
                return json.loads(e.read().decode())
            except Exception:
                return {"ok": False, "error": str(e)}
        except Exception as e:
            return {"ok": False, "error": str(e)}


# ── Helpers ────────────────────────────────────────────────────────

def _clean_slack_mentions(text: str) -> str:
    """Replace Slack mention format <@U123> with @user, <#C123> with #channel."""
    import re

    # User mentions: <@U123> → @user, <@U123|name> → @name
    def _replace_user(m: re.Match) -> str:
        if "|" in m.group(0):
            return "@" + m.group(0).split("|")[1].rstrip(">")
        return "@user"

    text = re.sub(r"<@[A-Z0-9]+(?:\|[^>]+)?>", _replace_user, text)

    # Channel mentions: <#C123> → #channel, <#C123|name> → #name
    def _replace_channel(m: re.Match) -> str:
        if "|" in m.group(0):
            return "#" + m.group(0).split("|")[1].rstrip(">")
        return "#channel"

    text = re.sub(r"<#[A-Z0-9]+(?:\|[^>]+)?>", _replace_channel, text)

    # Link format: <url> → url, <url|text> → text
    def _replace_link(m: re.Match) -> str:
        if "|" in m.group(0):
            return m.group(0).split("|")[1].rstrip(">")
        return m.group(0)[1:-1]

    text = re.sub(r"<(https?://[^>]+)(?:\|[^>]+)?>", _replace_link, text)

    # Special mentions: <!here>, <!channel>, <!everyone>
    text = text.replace("<!here>", "@here")
    text = text.replace("<!channel>", "@channel")
    text = text.replace("<!everyone>", "@everyone")

    return text


def _build_message_blocks(response: UnifiedResponse) -> list[dict[str, Any]]:
    """Build Slack Block Kit blocks from a UnifiedResponse."""
    blocks: list[dict[str, Any]] = []

    # Text section
    if response.text:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": response.text[:3000],
            },
        })

    # Evidence as context block
    if response.evidence:
        evidence_text = "\n".join(
            f"• {e.get('text', str(e))}" for e in response.evidence[:5]
        )
        if evidence_text:
            blocks.append({
                "type": "context",
                "elements": [{
                    "type": "mrkdwn",
                    "text": evidence_text[:2000],
                }],
            })

    # Confidence as context
    if response.confidence is not None:
        blocks.append({
            "type": "context",
            "elements": [{
                "type": "mrkdwn",
                "text": f"Confidence: {response.confidence:.0%}",
            }],
        })

    return blocks[:50]  # Slack limit: 50 blocks


def _build_media_block(
    media_url: str, media: list[dict[str, str]] | None = None
) -> dict[str, Any] | None:
    """Build an image block for media attachment."""
    url_lower = media_url.lower()
    if any(url_lower.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg")):
        alt_text = ""
        if media:
            alt_text = media[0].get("text", media[0].get("title", "image"))
        return {
            "type": "image",
            "image_url": media_url,
            "alt_text": alt_text or "attached image",
        }
    # Not an image — skip or return a link section
    return {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": f"📎 <{media_url}|Attachment>",
        },
    }
