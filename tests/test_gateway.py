"""Integration tests for Kairos framework."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

import json
import tempfile
from pathlib import Path
import pytest
from kairos.infra.knowledge.schema import KnowledgeSchema
from kairos.infra.knowledge.store import KnowledgeStore
from kairos.infra.rag.vector_store import VectorStore
from kairos.infra.rag.adapters import MarkdownAdapter, TextAdapter
from kairos.core.state import Case, Step, ThreadState, merge_artifacts
from kairos.core.middleware import MiddlewarePipeline
from kairos.middleware import (
    EvidenceTracker,
    ConfidenceScorer,
    ContextCompressor,
    SkillLoader,
    Skill,
)
from kairos.middleware.evidence import EvidenceTracker as ET
from kairos.infra.evidence.tracker import EvidenceDB
from kairos.chat.session import SessionStore
from kairos.tools.rag_search import rag_search, set_rag_store
from kairos.tools.knowledge_lookup import knowledge_lookup, set_knowledge_store
from kairos.tools.registry import get_all_tools, get_tool_schemas, execute_tool
from kairos.prompt.template import PromptBuilder
from kairos.agents.types import SubAgentType, BUILTIN_TYPES, GENERAL_PURPOSE, BASH, RESEARCH
from kairos.agents.factory import register_subagent_types, get_subagent_type
from kairos.memory.store import MemoryStore
from kairos.memory.middleware import MemoryMiddleware
from kairos.skills.manager import SkillManager, SkillStatus
from kairos.session.search import SessionSearch
from kairos.sandbox import (
    LocalSandbox, DockerSandbox, SSHSandbox,
    SandboxConfig, SandboxProvider, create_sandbox,
)
from kairos.gateway.protocol import (
    UnifiedMessage, UnifiedResponse, ContentBlock, ContentType,
    MessageRole, ConnectionState,
)
from kairos.gateway.adapters.base import CLIAdapter, TelegramAdapter, SlackAdapter
from kairos.training.recorder import TrajectoryRecorder, ToolContext
from kairos.training.env import (
    TrainingEnv, EnvironmentRegistry, RolloutRunner,
    reward_confidence, reward_success_rate, reward_evidence_quality, reward_file_creation,
)
from kairos.middleware.dangling import DanglingToolCallMiddleware
from kairos.middleware.subagent_limit import SubagentLimitMiddleware
from kairos.middleware.clarify import ClarificationMiddleware
from kairos.middleware.thread_data import ThreadDataMiddleware
from kairos.middleware.todo import TodoMiddleware
from kairos.middleware.title import TitleMiddleware
from kairos.middleware.uploads import UploadsMiddleware
from kairos.middleware.view_image import ViewImageMiddleware
from kairos.providers.credential import CredentialPool, Credential, RetryConfig
from kairos.middleware.llm_retry import LLMRetryMiddleware, ToolArgRepairMiddleware
from kairos.core.stateful_agent import StatefulAgent
from kairos.providers.base import ModelConfig


class TestGatewayAdaptersV2:
    def test_discord_init(self):
        from kairos.gateway.adapters import DiscordAdapter
        a = DiscordAdapter(bot_token="test")
        assert a.platform_name == "discord"

    def test_feishu_init(self):
        from kairos.gateway.adapters import FeishuAdapter
        a = FeishuAdapter(app_id="test", app_secret="secret")
        assert a.platform_name == "feishu"

    def test_whatsapp_init(self):
        from kairos.gateway.adapters import WhatsAppAdapter
        a = WhatsAppAdapter(phone_number_id="123", access_token="tok")
        assert a.platform_name == "whatsapp"

    def test_signal_init(self):
        from kairos.gateway.adapters import SignalAdapter
        a = SignalAdapter(sender_number="+1234567890")
        assert a.platform_name == "signal"

    def test_line_init(self):
        from kairos.gateway.adapters import LineAdapter
        a = LineAdapter(channel_access_token="tok")
        assert a.platform_name == "line"

    def test_matrix_init(self):
        from kairos.gateway.adapters import MatrixAdapter
        a = MatrixAdapter(homeserver="https://matrix.org", access_token="tok")
        assert a.platform_name == "matrix"

    def test_irc_init(self):
        from kairos.gateway.adapters import IRCAdapter
        a = IRCAdapter(server="irc.example.com", nickname="kairos-test")
        assert a.platform_name == "irc"

    def test_discord_translate(self):
        from kairos.gateway.adapters import DiscordAdapter
        a = DiscordAdapter()
        # Discord interaction payload format
        msg = a.translate_incoming({
            "id": "msg1",
            "type": 0,  # Not PING
            "channel_id": "C123",
            "guild_id": "G456",
            "author": {"id": "U789", "username": "buer"},
            "data": {"content": "hello kairos"},
        })
        assert msg.platform == "discord"
        assert "hello kairos" in msg.content[0].text
        assert msg.chat_id == "C123"

    def test_whatsapp_translate(self):
        from kairos.gateway.adapters import WhatsAppAdapter
        a = WhatsAppAdapter()
        msg = a.translate_incoming({
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [{
                            "id": "w1",
                            "from": "551199999999",
                            "type": "text",
                            "text": {"body": "hi from wa"},
                        }],
                        "contacts": [{"profile": {"name": "Test"}}],
                    }
                }]
            }]
        })
        assert msg.platform == "whatsapp"
        assert "hi from wa" in msg.content[0].text
        assert msg.chat_id == "551199999999"

    def test_line_translate(self):
        from kairos.gateway.adapters import LineAdapter
        a = LineAdapter()
        msg = a.translate_incoming({
            "events": [{
                "type": "message",
                "message": {"type": "text", "text": "hello line"},
                "source": {"userId": "U123"},
                "replyToken": "rtok",
                "webhookEventId": "evt1",
            }]
        })
        assert msg.platform == "line"
        assert "hello line" in msg.content[0].text
        assert msg.chat_id == "U123"

