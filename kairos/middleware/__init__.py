"""Middleware package — Kairos middleware pipeline (14 layers).

Layer ordering follows DeerFlow's dependency chain:

  1. ThreadData         — workspace dirs (must be first)
  2. Uploads            — file injection (depends on thread_data)
  3. DanglingToolCall   — fix broken tool calls (before any model call)
  4. SkillLoader        — load skills into context
  5. ContextCompressor  — summarize when tokens near limit
  6. Todo               — todo list persistence (after compression)
  7. Memory             — inject persistent memory
  8. ViewImage          — image injection for vision models (before model)
  9. EvidenceTracker    — record tool calls as evidence steps
 10. ConfidenceScorer   — evaluate output confidence
 11. SubagentLimit      — cap concurrent sub-agent calls (after model)
 12. Title              — auto-generate session title
 13. MemoryMiddleware   — submit to long-term memory queue (after title)
 14. Clarification      — intercept ask_user (MUST be last, may interrupt)
"""

from kairos.middleware.evidence import EvidenceTracker
from kairos.middleware.confidence import ConfidenceScorer
from kairos.middleware.compress import ContextCompressor
from kairos.middleware.skill_loader import SkillLoader, Skill
from kairos.middleware.dangling import DanglingToolCallMiddleware
from kairos.middleware.subagent_limit import SubagentLimitMiddleware
from kairos.middleware.clarify import ClarificationMiddleware
from kairos.middleware.thread_data import ThreadDataMiddleware
from kairos.middleware.todo import TodoMiddleware
from kairos.middleware.title import TitleMiddleware
from kairos.middleware.uploads import UploadsMiddleware
from kairos.middleware.view_image import ViewImageMiddleware

# Re-export MemoryMiddleware from memory package
from kairos.memory.middleware import MemoryMiddleware

__all__ = [
    "EvidenceTracker",
    "ConfidenceScorer",
    "ContextCompressor",
    "SkillLoader",
    "Skill",
    "DanglingToolCallMiddleware",
    "SubagentLimitMiddleware",
    "ClarificationMiddleware",
    "ThreadDataMiddleware",
    "TodoMiddleware",
    "TitleMiddleware",
    "UploadsMiddleware",
    "ViewImageMiddleware",
    "MemoryMiddleware",
]
