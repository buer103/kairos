"""Gateway package — multi-platform messaging and HTTP transport.

11 platform adapters:
    CLI, Telegram, WeChat, Slack, Discord, Feishu, WhatsApp,
    Signal, Line, Matrix, IRC
"""

from kairos.gateway.protocol import (
    UnifiedMessage,
    UnifiedResponse,
    ContentBlock,
    ContentType,
    MessageRole,
    ConnectionState,
)
from kairos.gateway.server import GatewayServer
from kairos.gateway.adapters import (
    PlatformAdapter,
    CLIAdapter,
    TelegramAdapter,
    WeChatAdapter,
    SlackAdapter,
    DiscordAdapter,
    FeishuAdapter,
    WhatsAppAdapter,
    SignalAdapter,
    LineAdapter,
    MatrixAdapter,
    IRCAdapter,
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
    "DiscordAdapter",
    "FeishuAdapter",
    "WhatsAppAdapter",
    "SignalAdapter",
    "LineAdapter",
    "MatrixAdapter",
    "IRCAdapter",
]
