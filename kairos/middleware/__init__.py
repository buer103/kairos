"""Middleware package — Kairos middleware pipeline (16 layers).

Layer ordering follows DeerFlow's dependency chain + Kairos additions:

  1. ThreadData         — workspace dirs
  2. Uploads            — file injection
  3. DanglingToolCall   — fix broken tool calls
  4. SkillLoader        — load skills
  5. ContextCompressor  — token budget
  6. Todo               — plan mode persistence
  7. Memory             — persistent memory
  8. ViewImage          — vision support
  9. EvidenceTracker    — evidence chain (Kairos)
 10. ToolArgRepair      — repair broken JSON args (Kairos)
 11. ConfidenceScorer   — output quality (Kairos)
 12. LLMRetry           — retry with credential rotation (Kairos)
 13. SubagentLimit      — cap concurrent sub-agents
 14. Title              — auto-generate title
 15. MemoryMiddleware   — submit to memory queue
 16. Clarification      — intercept ask_user (MUST be last)
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
from kairos.middleware.llm_retry import LLMRetryMiddleware, ToolArgRepairMiddleware
from kairos.middleware.logging_mw import LoggingMiddleware

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
    "LLMRetryMiddleware",
    "ToolArgRepairMiddleware",
    "LoggingMiddleware",
    "MemoryMiddleware",
]
