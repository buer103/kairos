"""Platform adapter implementations — 11 platform adapters for Kairos Gateway.

Each adapter translates platform-native messages to Kairos' UnifiedMessage
and translates UnifiedResponse back to the platform's native format.

Adapters:
    CLI         — stdin/stdout (always available)
    Telegram    — Bot API (httpx)
    WeChat      — Official Account / Customer Service API
    Slack       — Web API (slack-sdk optional)
    Discord     — Gateway / Webhook
    Feishu      — Lark Open API
    WhatsApp    — Meta Cloud API / Twilio
    Signal      — signal-cli REST wrapper
    Line        — Messaging API
    Matrix      — Matrix Client-Server API
    IRC         — IRC protocol
"""

from kairos.gateway.adapters.base import (
    PlatformAdapter,
    CLIAdapter,
    TelegramAdapter,
    WeChatAdapter,
    SlackAdapter,
)
from kairos.gateway.adapters.discord import DiscordAdapter
from kairos.gateway.adapters.feishu import FeishuAdapter
from kairos.gateway.adapters.whatsapp import WhatsAppAdapter
from kairos.gateway.adapters.signal import SignalAdapter
from kairos.gateway.adapters.line import LineAdapter
from kairos.gateway.adapters.matrix import MatrixAdapter
from kairos.gateway.adapters.irc import IRCAdapter

__all__ = [
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
