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
from kairos.gateway.adapters.base import CLIAdapter
from kairos.gateway.adapters import TelegramAdapter, SlackAdapter
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


class TestCronScheduler:
    def test_register_and_list(self, tmp_path):
        from kairos.cron import CronScheduler, Job, CronSchedule
        db = tmp_path / "cron.db"
        s = CronScheduler(db_path=str(db))
        j = Job(name="daily", schedule=CronSchedule.daily_at(9, 0))
        s.register(j)
        jobs = s.list()
        assert len(jobs) == 1
        assert jobs[0].name == "daily"

    def test_pause_resume(self, tmp_path):
        from kairos.cron import CronScheduler, Job, CronSchedule, JobStatus
        db = tmp_path / "cron.db"
        s = CronScheduler(db_path=str(db))
        j = s.register(Job(name="test"))
        assert s.pause(j.id).status == JobStatus.PAUSED
        assert s.resume(j.id).status == JobStatus.PENDING

    def test_cancel(self, tmp_path):
        from kairos.cron import CronScheduler, Job, CronSchedule, JobStatus
        db = tmp_path / "cron.db"
        s = CronScheduler(db_path=str(db))
        j = s.register(Job(name="test"))
        assert s.cancel(j.id).status == JobStatus.CANCELLED

    def test_remove(self, tmp_path):
        from kairos.cron import CronScheduler, Job
        db = tmp_path / "cron.db"
        s = CronScheduler(db_path=str(db))
        j = s.register(Job(name="removable"))
        s.remove(j.id)
        assert len(s.list()) == 0

    def test_schedule_matches(self):
        from kairos.cron import CronSchedule
        from datetime import datetime, timezone
        sched = CronSchedule.daily_at(9, 0)
        dt = datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc)
        assert sched.matches(dt)
        assert not sched.matches(datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc))

    def test_schedule_weekly(self):
        from kairos.cron import CronSchedule
        from datetime import datetime, timezone
        sched = CronSchedule.weekly_on(0, 9, 0)  # Monday 9:00
        mon = datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc)  # Monday
        tue = datetime(2026, 1, 6, 9, 0, tzinfo=timezone.utc)  # Tuesday
        assert sched.matches(mon)
        assert not sched.matches(tue)

    def test_tick_fires_due_jobs(self, tmp_path):
        from kairos.cron import CronScheduler, Job, CronSchedule
        from datetime import datetime, timezone
        db = tmp_path / "cron.db"
        s = CronScheduler(db_path=str(db))
        now = datetime.now(timezone.utc)
        sched = CronSchedule(minute=[now.minute], hour=[now.hour])
        j = s.register(Job(name="immediate", schedule=sched))
        # Tick fires jobs that match current time
        fired = s.tick()
        assert len(fired) >= 1
        updated = s.get(j.id)
        assert updated.run_count >= 1

    def test_repeat_limit(self, tmp_path):
        from kairos.cron import CronScheduler, Job, CronSchedule
        from datetime import datetime, timezone
        db = tmp_path / "cron.db"
        s = CronScheduler(db_path=str(db))
        now = datetime.now(timezone.utc)
        sched = CronSchedule(minute=[now.minute], hour=[now.hour])
        j = s.register(Job(name="limited", schedule=sched, repeat=1))
        s.tick()
        j = s.get(j.id)
        assert j.run_count == 1
        assert j.status.value == "done"
        # Tick again — should not re-fire (repeat=1 reached)
        fired = s.tick()
        assert len(fired) == 0

    def test_next_fire(self):
        from kairos.cron import CronSchedule
        from datetime import datetime, timezone, timedelta
        sched = CronSchedule.daily_at(12, 0)
        now = datetime(2026, 1, 5, 8, 0, tzinfo=timezone.utc)
        nxt = sched.next_fire(now)
        assert nxt.hour == 12
        assert nxt.day == 5
        assert nxt > now



