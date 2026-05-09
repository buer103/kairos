"""Platform adapter base class — translate platform messages to UnifiedMessage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from kairos.gateway.protocol import (
    UnifiedMessage,
    UnifiedResponse,
    ContentBlock,
    ContentType,
    MessageRole,
    ConnectionState,
)


class PlatformAdapter(ABC):
    """Base class for platform adapters.

    Each platform (Telegram, Discord, WeChat, Slack, etc.) subclasses this
    to translate its native API messages into Kairos' UnifiedMessage format,
    and translate UnifiedResponse back to the platform's native format.
    """

    platform_name: str = "unknown"

    def __init__(self):
        self._state = ConnectionState.DISCONNECTED

    @property
    def state(self) -> ConnectionState:
        return self._state

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the platform."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the platform."""
        ...

    @abstractmethod
    async def send(self, chat_id: str, response: UnifiedResponse) -> bool:
        """Send a response to a specific chat."""
        ...

    @abstractmethod
    async def receive(self) -> UnifiedMessage | None:
        """Receive the next message from the platform."""
        ...

    def translate_incoming(self, raw: dict[str, Any]) -> UnifiedMessage:
        """Convert a platform-specific message to UnifiedMessage.

        Override in subclasses for platform-specific parsing.
        """
        import uuid
        return UnifiedMessage(
            id=str(uuid.uuid4())[:12],
            role=raw.get("role", "user"),
            content=raw.get("content", []),
            platform=self.platform_name,
            chat_id=raw.get("chat_id", ""),
            thread_id=raw.get("thread_id", ""),
            sender_id=raw.get("sender_id", ""),
            sender_name=raw.get("sender_name", ""),
        )

    def translate_outgoing(self, response: UnifiedResponse) -> dict[str, Any]:
        """Convert a UnifiedResponse to the platform-specific format.

        Override in subclasses for platform-specific formatting.
        """
        return response.to_dict()


class CLIAdapter(PlatformAdapter):
    """Built-in CLI adapter (stdin/stdout). Always available."""

    platform_name = "cli"

    def __init__(self):
        super().__init__()
        self._state = ConnectionState.CONNECTED

    async def connect(self) -> bool:
        self._state = ConnectionState.CONNECTED
        return True

    async def disconnect(self) -> None:
        self._state = ConnectionState.DISCONNECTED

    async def send(self, chat_id: str, response: UnifiedResponse) -> bool:
        print(f"\n🤖 Kairos:\n{response.text}")
        if response.confidence is not None:
            print(f"   (confidence: {response.confidence:.2f})")
        return True

    async def receive(self) -> UnifiedMessage | None:
        try:
            text = input("🤖 Kairos> ").strip()
            if not text:
                return None
            import uuid
            return UnifiedMessage(
                id=str(uuid.uuid4())[:12],
                role=MessageRole.USER,
                content=[ContentBlock.text_block(text)],
                platform=self.platform_name,
                chat_id="cli-session",
            )
        except (KeyboardInterrupt, EOFError):
            return None

# Adapter imports are at the package level (kairos.gateway.adapters)
# TelegramAdapter, WeChatAdapter, SlackAdapter are defined in their own files:
#   kairos/gateway/adapters/telegram.py
#   kairos/gateway/adapters/wechat.py
#   kairos/gateway/adapters/slack.py
