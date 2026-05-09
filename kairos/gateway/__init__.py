"""Gateway package — multi-platform messaging, HTTP transport, and device pairing.

12 platform adapters:
    CLI, Telegram, WeChat, Slack, Discord, Feishu, WhatsApp,
    Signal, Line, Matrix, IRC (11) + Generic

Core modules:
    - protocol.py: UnifiedMessage / UnifiedResponse types
    - server.py: HTTP+SSE API server with session management
    - manager.py: Central adapter lifecycle + routing + health
    - webhook.py: Inbound webhook server with per-platform verification
    - pairing.py: Device pairing (QR code + verification code)
    - ratelimit.py: Sliding-window rate limiting
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
from kairos.gateway.manager import GatewayManager, AdapterHealth, RouteResult
from kairos.gateway.webhook import WebhookServer
from kairos.gateway.pairing import PairingManager, PairingRequest, PairingState
from kairos.gateway.ratelimit import RateLimiter, MultiTierLimiter
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
    # Protocol
    "UnifiedMessage",
    "UnifiedResponse",
    "ContentBlock",
    "ContentType",
    "MessageRole",
    "ConnectionState",
    # Core
    "GatewayServer",
    "GatewayManager",
    "AdapterHealth",
    "RouteResult",
    "WebhookServer",
    "PairingManager",
    "PairingRequest",
    "PairingState",
    "RateLimiter",
    "MultiTierLimiter",
    # Adapters
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
