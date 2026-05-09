"""Platform adapter implementations — 11 platform adapters for Kairos Gateway.

Each adapter translates platform-native messages to Kairos' UnifiedMessage
and translates UnifiedResponse back to the platform's native format.

Adapters:
    CLI         — stdin/stdout (always available, in base.py)
    Telegram    — Bot API (telegram.py, full REST impl)
    WeChat      — Official Account API (wechat.py, XML + signature)
    Slack       — Web API (slack.py, Events API + block kit)
    Discord     — Gateway/Webhook (discord.py)
    Feishu      — Lark Open API (feishu.py)
    WhatsApp    — Meta Cloud API (whatsapp.py)
    Signal      — signal-cli REST (signal.py)
    Line        — Messaging API (line.py)
    Matrix      — Client-Server API (matrix.py)
    IRC         — IRC protocol (irc.py)
"""

from kairos.gateway.adapters.base import PlatformAdapter, CLIAdapter
from kairos.gateway.adapters.telegram import TelegramAdapter
from kairos.gateway.adapters.wechat import WeChatAdapter
from kairos.gateway.adapters.slack import SlackAdapter
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
