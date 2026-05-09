"""Gateway protocol — unified message types for multi-platform agent access.

Every platform adapter (CLI, HTTP, Telegram, Discord, etc.) translates
its native message format into these unified types, and the Agent responds
with a unified response that the adapter translates back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ContentType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    FILE = "file"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass
class ContentBlock:
    """A single content block within a message."""
    type: ContentType = ContentType.TEXT
    text: str = ""
    url: str = ""
    mime_type: str = ""

    def to_dict(self) -> dict:
        d = {"type": self.type.value}
        if self.text:
            d["text"] = self.text
        if self.url:
            d["url"] = self.url
        if self.mime_type:
            d["mime_type"] = self.mime_type
        return d

    @classmethod
    def text_block(cls, text: str) -> ContentBlock:
        return cls(type=ContentType.TEXT, text=text)

    @classmethod
    def image_block(cls, url: str) -> ContentBlock:
        return cls(type=ContentType.IMAGE, url=url)


@dataclass
class UnifiedMessage:
    """A message from any platform, normalized to Kairos' internal format."""

    id: str
    role: MessageRole
    content: list[ContentBlock] = field(default_factory=list)
    platform: str = ""  # e.g. "telegram", "discord", "wechat", "cli", "http"
    chat_id: str = ""
    thread_id: str = ""
    sender_id: str = ""
    sender_name: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    raw_payload: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Extract plain text from content blocks."""
        return " ".join(b.text for b in self.content if b.type == ContentType.TEXT)

    @property
    def has_media(self) -> bool:
        return any(b.type != ContentType.TEXT for b in self.content)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "role": self.role.value,
            "content": [b.to_dict() for b in self.content],
            "platform": self.platform,
            "chat_id": self.chat_id,
            "thread_id": self.thread_id,
            "sender_id": self.sender_id,
            "sender_name": self.sender_name,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_text(cls, text: str, platform: str = "cli", chat_id: str = "") -> UnifiedMessage:
        """Quick constructor for text-only messages."""
        import uuid
        return cls(
            id=str(uuid.uuid4())[:12],
            role=MessageRole.USER,
            content=[ContentBlock.text_block(text)],
            platform=platform,
            chat_id=chat_id,
        )


@dataclass
class UnifiedResponse:
    """Agent's response, to be translated back to the platform's native format."""

    text: str
    media: list[dict[str, str]] = field(default_factory=list)  # [{type, url/path}]
    confidence: float | None = None
    evidence: list[dict] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "media": self.media,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "metadata": self.metadata,
        }


class ConnectionState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
