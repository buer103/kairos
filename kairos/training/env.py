"""Training environment — RL training pipeline for Kairos Agent.

Implements:
  - Environment registry for multi-domain training
  - Rollout runner with tool context
  - Reward function framework
  - Atropos-compatible trajectory export
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from kairos.core.loop import Agent
from kairos.gateway.protocol import UnifiedMessage
from kairos.training.recorder import TrajectoryRecorder, ToolContext

# Type alias for reward functions
RewardFn = Callable[[dict[str, Any], ToolContext], float]


@dataclass
class TrainingEnv:
    """A training environment defining a domain + reward function.

    Usage:
        def reward_diagnosis(result, ctx):
            score = 0.0
            if result.get("confidence", 0) > 0.8:
                score += 1.0
            if ctx.grep_output("root cause"):
                score += 1.0
            return score

        env = TrainingEnv(
            name="vehicle-diagnosis",
            prompt_template="Diagnose the issue in {log_file}",
            reward_fn=reward_diagnosis,
        )
    """

    name: str
    description: str = ""
    prompt_template: str = ""
    reward_fn: RewardFn = field(default=lambda r, c: 0.0)
    tools: list[str] = field(default_factory=list)
    max_turns: int = 20
    metadata: dict[str, Any] = field(default_factory=dict)

    def format_prompt(self, **kwargs) -> str:
        """Format the prompt template with variables."""
        return self.prompt_template.format(**kwargs) if self.prompt_template else ""

    def evaluate(self, result: dict[str, Any], context: ToolContext) -> float:
        """Compute the reward for this rollout."""
        try:
            return self.reward_fn(result, context)
        except Exception:
            return 0.0


class EnvironmentRegistry:
    """Registry of training environments."""

    def __init__(self):
        self._envs: dict[str, TrainingEnv] = {}

    def register(self, env: TrainingEnv) -> None:
        self._envs[env.name] = env

    def get(self, name: str) -> TrainingEnv | None:
        return self._envs.get(name)

    def list_envs(self) -> list[str]:
        return list(self._envs.keys())


class RolloutRunner:
    """Run multi-turn rollouts of the agent in a training environment.

    Records trajectories and computes rewards via ToolContext.
    """

    def __init__(
        self,
        agent: Agent,
        recorder: TrajectoryRecorder | None = None,
        workdir: str | Path | None = None,
    ):
        self.agent = agent
        self.recorder = recorder or TrajectoryRecorder()
        self.workdir = Path(workdir or Path.cwd())

    def run(
        self,
        prompt: str,
        reward_fn: RewardFn | None = None,
        tools: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Run a single rollout and return the result with reward.

        Args:
            prompt: the task prompt
            reward_fn: optional reward function for this specific rollout
            tools: optional tool schemas

        Returns:
            dict with: content, confidence, evidence, reward, trajectory_id
        """
        ctx = ToolContext(workdir=self.workdir)
        ctx.snapshot_before()

        import time
        start = time.time()
        result = self.agent.run(prompt)
        duration_ms = (time.time() - start) * 1000

        ctx.snapshot_after()

        # Record trajectory
        messages = self.agent._prompt_builder.build() if hasattr(self.agent, '_prompt_builder') else ""
        # We record the conversation from the agent's state
        self.recorder.record(
            messages=[{"role": "user", "content": prompt}, {"role": "assistant", "content": result.get("content", "")}],
            tools=tools,
            metadata={
                "confidence": result.get("confidence"),
                "evidence": result.get("evidence", []),
                "duration_ms": duration_ms,
            },
        )

        reward = (reward_fn or (lambda r, c: 0.0))(result, ctx)

        return {
            "content": result.get("content", ""),
            "confidence": result.get("confidence"),
            "evidence": result.get("evidence", []),
            "reward": reward,
            "duration_ms": duration_ms,
            "trajectory_id": self.recorder.total_recorded() - 1,
        }

    def run_batch(
        self,
        prompts: list[str],
        reward_fn: RewardFn | None = None,
    ) -> list[dict[str, Any]]:
        """Run multiple rollouts and return results."""
        return [self.run(prompt, reward_fn=reward_fn) for prompt in prompts]

    def flush_trajectories(self) -> Path:
        """Write recorded trajectories to disk."""
        return self.recorder.flush()


# ── Built-in Reward Functions ──────────────────────────────────


def reward_confidence(result: dict[str, Any], ctx: ToolContext) -> float:
    """Reward based on confidence score."""
    conf = result.get("confidence")
    if conf is None:
        return 0.0
    return float(conf)


def reward_success_rate(result: dict[str, Any], ctx: ToolContext) -> float:
    """Reward based on content length (non-empty = success)."""
    content = result.get("content", "")
    return 1.0 if content and len(content) > 10 else 0.0


def reward_evidence_quality(result: dict[str, Any], ctx: ToolContext) -> float:
    """Reward based on number of evidence steps."""
    evidence = result.get("evidence", [])
    return min(len(evidence) / 5.0, 1.0)  # Max at 5+ steps


def reward_file_creation(result: dict[str, Any], ctx: ToolContext) -> float:
    """Reward based on whether the agent created any expected files."""
    # Check if any terminal output mentions file creation
    if ctx.grep_output("created") or ctx.grep_output("written") or ctx.grep_output("saved"):
        return 1.0
    # Check if any files were actually created
    before = ctx._files_before
    after = ctx._files_after
    new_files = set(after.keys()) - set(before.keys())
    return 1.0 if new_files else 0.0
