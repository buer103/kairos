"""
Matrix Client-Server API adapter.

Docs: https://spec.matrix.org/latest/client-server-api/
No external library required — pure REST API over urllib.
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


class MatrixAdapter(PlatformAdapter):
    """Matrix adapter — send/receive via Matrix C-S API."""

    platform_name = "matrix"

    def __init__(
        self,
        homeserver: str = "https://matrix.org",
        user_id: str = "",
        access_token: str = "",
        room_id: str = "",
    ):
        """
        Args:
            homeserver: Matrix homeserver URL
            user_id: Full MXID (e.g., @bot:matrix.org)
            access_token: Matrix access token
            room_id: Default room to send/receive from
        """
        super().__init__()
        self._homeserver = homeserver.rstrip("/")
        self._user_id = user_id
        self._token = access_token
        self._room_id = room_id
        self._txn_id = 0
        self._since = ""  # Sync token for /sync polling

    async def connect(self) -> bool:
        if not self._token:
            self._state = ConnectionState.ERROR
            return False
        try:
            # Verify connection — whoami
            url = f"{self._homeserver}/_matrix/client/v3/account/whoami"
            req = urllib.request.Request(
                url,
                headers={"Authorization": f"Bearer {self._token}"},
            )
            resp = urllib.request.urlopen(req, timeout=10)
            if resp.status == 200:
                data = json.loads(resp.read().decode())
                self._user_id = data.get("user_id", self._user_id)
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
        """Send message to a Matrix room."""
        if not self._token:
            return False

        room = chat_id or self._room_id
        if not room:
            return False

        try:
            self._txn_id += 1
            url = f"{self._homeserver}/_matrix/client/v3/rooms/{room}/send/m.room.message/{self._txn_id}"

            body: dict[str, Any] = {
                "msgtype": "m.text",
                "body": response.text,
            }

            # Format as markdown if no media
            if not response.media_url:
                body["format"] = "org.matrix.custom.html"
                body["formatted_body"] = (
                    f"<p>{_escape_html(response.text).replace(chr(10), '<br>')}</p>"
                )
            else:
                # Image message
                body["msgtype"] = "m.image"
                body["url"] = response.media_url
                body["body"] = response.text or "Image"

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
        """Poll /sync for new messages."""
        if not self._token:
            return None

        try:
            params = "?timeout=5000"  # 5-second long poll
            if self._since:
                params += f"&since={self._since}"

            url = f"{self._homeserver}/_matrix/client/v3/sync{params}"
            req = urllib.request.Request(
                url,
                headers={"Authorization": f"Bearer {self._token}"},
            )
            resp = urllib.request.urlopen(req, timeout=10)
            data = json.loads(resp.read().decode())

            self._since = data.get("next_batch", self._since)

            # Extract the first new text message
            rooms = data.get("rooms", {}).get("join", {})
            for room_id, room_data in rooms.items():
                timeline = room_data.get("timeline", {}).get("events", [])
                for event in timeline:
                    if event.get("type") == "m.room.message":
                        sender = event.get("sender", "")
                        # Skip own messages
                        if sender == self._user_id:
                            continue
                        return self.translate_incoming({
                            **event,
                            "room_id": room_id,
                        })

            return None
        except Exception:
            return None

    def translate_incoming(self, raw: dict[str, Any]) -> UnifiedMessage:
        """Parse a Matrix m.room.message event."""
        content = raw.get("content", {})
        msgtype = content.get("msgtype", "m.text")

        text = content.get("body", "")
        if msgtype == "m.image":
            text = f"[image] {text}"
        elif msgtype == "m.file":
            text = f"[file] {text}"
        elif msgtype == "m.audio":
            text = f"[audio] {text}"
        elif msgtype == "m.video":
            text = f"[video] {text}"

        return UnifiedMessage(
            id=raw.get("event_id", ""),
            role=MessageRole.USER,
            content=[ContentBlock.text_block(text)],
            platform="matrix",
            chat_id=raw.get("room_id", ""),
            thread_id=content.get("m.relates_to", {}).get("event_id", ""),
            sender_id=raw.get("sender", ""),
            sender_name=raw.get("sender", "").split(":")[0].lstrip("@") if raw.get("sender") else "",
        )


def _escape_html(text: str) -> str:
    """Escape HTML entities for Matrix formatted_body."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
