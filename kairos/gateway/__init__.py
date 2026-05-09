"""Gateway package — multi-platform messaging and HTTP transport."""

from kairos.gateway.protocol import (
    UnifiedMessage,
    UnifiedResponse,
    ContentBlock,
    ContentType,
    MessageRole,
    ConnectionState,
)
from kairos.gateway.server import GatewayServer
from kairos.gateway.adapters.base import (
    PlatformAdapter,
    CLIAdapter,
    TelegramAdapter,
    WeChatAdapter,
    SlackAdapter,
)

__all__ = [
    "UnifiedMessage",
    "UnifiedResponse",
    "ContentBlock",
    "ContentType",
    "MessageRole",
    "ConnectionState",
    "GatewayServer",
    "PlatformAdapter",
    "CLIAdapter",
    "TelegramAdapter",
    "WeChatAdapter",
    "SlackAdapter",
]
