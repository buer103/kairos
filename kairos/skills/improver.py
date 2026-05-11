r"""Skill self-improvement loop — autonomous skill creation and refinement.

Hermes-compatible: the agent reviews its own performance, identifies gaps,
and creates or updates skills without human intervention.

Workflow:
  1. REVIEW — analyze session history for patterns, errors, reusable workflows
  2. IDENTIFY — detect recurring tool sequences that could become skills
  3. GENERATE — produce SKILL.md content from observed patterns
  4. VALIDATE — check syntax, completeness, safety of new skills
  5. INSTALL — write to skills directory and trigger re-scan

Usage:
    improver = SkillImprover(skill_manager, model_provider)

    # After a session, review for improvements
    suggestions = improver.review_session(messages, tools_used)

    # Auto-create a skill from a suggestion
    if suggestions:
        improver.create_skill(suggestions[0])

    # Or run full self-improvement cycle
    result = improver.self_improve(messages, feedback="The agent forgot to...")
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("kairos.skills.improver")

# ═══════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════


@dataclass
class ImprovementSuggestion:
    """A concrete suggestion for skill improvement."""

    type: str  # "new_skill" | "update_skill" | "deprecate"
    title: str
    description: str
    pattern: list[dict]  # Example tool call sequence
    tools_used: list[str]
    confidence: float  # 0.0 - 1.0
    suggested_skill_name: str = ""
    suggested_steps: list[str] = field(default_factory=list)
    source: str = ""  # Where the pattern was observed


@dataclass
class SelfImproveResult:
    """Result of a self-improvement cycle."""

    suggestions: list[ImprovementSuggestion] = field(default_factory=list)
    skills_created: list[str] = field(default_factory=list)
    skills_updated: list[str] = field(default_factory=list)
    skills_skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    duration_ms: float = 0.0

    @property
    def total_changes(self) -> int:
        return len(self.skills_created) + len(self.skills_updated)


# ═══════════════════════════════════════════════════════════
# Skill Improver
# ═══════════════════════════════════════════════════════════


class SkillImprover:
    """Autonomous skill self-improvement loop.

    The agent analyzes its own conversation history to:
    - Identify recurring tool-call patterns worth capturing as skills
    - Detect errors or inefficiencies that could be prevented by a skill
    - Generate SKILL.md content from observed examples
    - Create or update skills in the manager
    """

    # Minimum number of occurrences to consider a pattern "recurring"
    MIN_PATTERN_OCCURRENCES = 2

    # Minimum sequence length for a reusable workflow
    MIN_SEQUENCE_LENGTH = 2

    # Default skill template
    SKILL_TEMPLATE = """---
name: {name}
description: "{description}"
version: 1.0.0
created: {created}
tools: [{tools}]
confidence: {confidence:.2f}
source: auto-generated
---

# {title}

{description}

## When to use

{when_to_use}

## Steps

{steps}

## Pitfalls

- This skill was auto-generated. Review before relying on it.
- If incorrect, the agent will improve it in future sessions.
"""

    def __init__(
        self,
        skill_manager: Any = None,
        model_provider: Any = None,
        skills_dir: str | Path | None = None,
        auto_install: bool = True,
        max_suggestions_per_cycle: int = 3,
    ):
        """Initialize the skill improver.

        Args:
            skill_manager: SkillManager instance for skill CRUD.
            model_provider: Optional ModelProvider for LLM-based improvement.
            skills_dir: Directory where skills are stored.
            auto_install: Automatically install created skills.
            max_suggestions_per_cycle: Cap on suggestions per self_improve().
        """
        self._manager = skill_manager
        self._model = model_provider
        self._skills_dir = Path(skills_dir) if skills_dir else None
        self._auto_install = auto_install
        self._max_suggestions = max_suggestions_per_cycle

        # State
        self._pattern_memory: dict[str, int] = {}  # pattern_hash → occurrence count
        self._last_review_time: float = 0.0

    # ── Public API ───────────────────────────────────────────────

    def review_session(
        self,
        messages: list[dict],
        tools_used: list[str] | None = None,
        feedback: str = "",
    ) -> list[ImprovementSuggestion]:
        """Analyze a session and generate improvement suggestions.

        Args:
            messages: Full conversation history (OpenAI format).
            tools_used: List of tool names used in this session.
            feedback: Optional human feedback about the session.

        Returns:
            List of concrete improvement suggestions, sorted by confidence.
        """
        suggestions: list[ImprovementSuggestion] = []

        # 1. Detect recurring tool call patterns
        pattern_suggestions = self._detect_recurring_patterns(messages, tools_used or [])
        suggestions.extend(pattern_suggestions)

        # 2. Detect error-recovery patterns (tools called after errors)
        recovery_suggestions = self._detect_error_recovery_patterns(messages)
        suggestions.extend(recovery_suggestions)

        # 3. Process explicit feedback
        if feedback:
            feedback_suggestions = self._process_feedback(feedback, messages)
            suggestions.extend(feedback_suggestions)

        # 4. Score and deduplicate
        suggestions = self._deduplicate_suggestions(suggestions)
        suggestions.sort(key=lambda s: s.confidence, reverse=True)

        # 5. Cap
        suggestions = suggestions[:self._max_suggestions]

        self._last_review_time = time.time()
        return suggestions

    def create_skill(self, suggestion: ImprovementSuggestion) -> str | None:
        """Create a new skill from a suggestion. Returns skill name or None."""
        if suggestion.type != "new_skill":
            logger.debug("Skipping non-new-skill suggestion: %s", suggestion.type)
            return None

        name = suggestion.suggested_skill_name or self._generate_skill_name(suggestion)
        content = self._render_skill_md(suggestion, name)

        # Validate
        errors = self._validate_skill_content(content)
        if errors:
            logger.warning("Skill validation failed for %s: %s", name, errors)
            return None

        # Install
        if self._auto_install:
            self._write_skill_file(name, content)
            # Trigger re-scan if manager available
            if self._manager and hasattr(self._manager, "scan"):
                self._manager.scan()
            logger.info("Created skill: %s", name)

        return name

    def update_skill(
        self,
        name: str,
        feedback: str,
        corrected_steps: list[str] | None = None,
    ) -> bool:
        """Update an existing skill based on feedback."""
        if not self._manager:
            return False

        current = self._get_skill_content(name)
        if current is None:
            logger.warning("Skill not found for update: %s", name)
            return False

        # Merge feedback into skill content
        improved = self._merge_feedback(current, feedback, corrected_steps)

        self._write_skill_file(name, improved)
        if hasattr(self._manager, "scan"):
            self._manager.scan()
        logger.info("Updated skill: %s", name)
        return True

    def self_improve(
        self,
        messages: list[dict],
        feedback: str = "",
        dry_run: bool = False,
    ) -> SelfImproveResult:
        """Run a full self-improvement cycle.

        1. Review session → suggestions
        2. For each suggestion → create or update skill
        3. Return results

        Args:
            messages: Full conversation history.
            feedback: Optional human feedback.
            dry_run: If True, generate suggestions without installing.

        Returns:
            SelfImproveResult with created/updated/error counts.
        """
        start = time.time()
        result = SelfImproveResult()

        # Step 1: Review
        tools_used = self._extract_tools_from_messages(messages)
        result.suggestions = self.review_session(messages, tools_used, feedback)

        if dry_run:
            result.duration_ms = (time.time() - start) * 1000
            return result

        # Step 2: Act on suggestions
        for suggestion in result.suggestions:
            try:
                if suggestion.type == "new_skill":
                    name = self.create_skill(suggestion)
                    if name:
                        result.skills_created.append(name)
                    else:
                        result.skills_skipped.append(suggestion.title)
                elif suggestion.type == "update_skill":
                    ok = self.update_skill(
                        suggestion.suggested_skill_name,
                        suggestion.description,
                        suggestion.suggested_steps,
                    )
                    if ok:
                        result.skills_updated.append(suggestion.suggested_skill_name)
                    else:
                        result.skills_skipped.append(suggestion.suggested_skill_name)
            except Exception as e:
                logger.error("Skill improvement error: %s", e)
                result.errors.append(str(e))

        result.duration_ms = (time.time() - start) * 1000
        return result

    # ── Pattern Detection ───────────────────────────────────────

    def _detect_recurring_patterns(
        self,
        messages: list[dict],
        tools_used: list[str],
    ) -> list[ImprovementSuggestion]:
        """Find tool-call sequences that repeat across conversations."""
        suggestions: list[ImprovementSuggestion] = []

        # Extract tool call sequences from assistant messages
        sequences = self._extract_tool_sequences(messages)

        # Group by normalized sequence signature
        signature_counts: dict[str, list[list[dict]]] = {}
        for seq in sequences:
            sig = self._sequence_signature(seq)
            if sig not in signature_counts:
                signature_counts[sig] = []
            signature_counts[sig].append(seq)

        # Filter to recurring patterns
        for sig, occurrences in signature_counts.items():
            if len(occurrences) >= self.MIN_PATTERN_OCCURRENCES:
                # Check if sequence is substantial enough
                if len(occurrences[0]) >= self.MIN_SEQUENCE_LENGTH:
                    suggestion = self._suggestion_from_sequence(
                        sig, occurrences, tools_used
                    )
                    if suggestion:
                        suggestions.append(suggestion)

        return suggestions

    def _detect_error_recovery_patterns(
        self,
        messages: list[dict],
    ) -> list[ImprovementSuggestion]:
        """Find sequences where tools were called after errors."""
        suggestions: list[ImprovementSuggestion] = []

        for i, msg in enumerate(messages):
            if msg.get("role") != "tool":
                continue
            content = str(msg.get("content", ""))
            # Check if tool result contains an error
            if "error" in content.lower() or "failed" in content.lower():
                # Look at what happened after the error
                recovery_sequence = []
                for j in range(i + 1, min(i + 5, len(messages))):
                    m = messages[j]
                    if m.get("role") == "assistant" and m.get("tool_calls"):
                        for tc in m["tool_calls"]:
                            recovery_sequence.append({
                                "tool": tc.get("function", {}).get("name", "?"),
                                "purpose": "error_recovery",
                            })
                if recovery_sequence:
                    suggestions.append(ImprovementSuggestion(
                        type="new_skill",
                        title=f"Error recovery: {content[:60]}",
                        description=f"After error '{content[:100]}', the agent recovered with: "
                                    + ", ".join(r["tool"] for r in recovery_sequence),
                        pattern=recovery_sequence,
                        tools_used=[r["tool"] for r in recovery_sequence],
                        confidence=0.6,
                        suggested_skill_name=self._slugify(f"error-recovery-{content[:30]}"),
                        source="error_recovery",
                    ))

        return suggestions

    def _process_feedback(
        self,
        feedback: str,
        messages: list[dict],
    ) -> list[ImprovementSuggestion]:
        """Generate suggestions from explicit human feedback."""
        if not feedback.strip():
            return []

        # Simple keyword-based suggestion generation
        suggestion = ImprovementSuggestion(
            type="new_skill",
            title=f"Feedback: {feedback[:60]}",
            description=feedback,
            pattern=[],
            tools_used=[],
            confidence=0.8,  # Human feedback = high confidence
            suggested_skill_name=self._slugify(f"improvement-{feedback[:40]}"),
            source="human_feedback",
        )
        return [suggestion]

    # ── Internal ─────────────────────────────────────────────────

    def _extract_tool_sequences(self, messages: list[dict]) -> list[list[dict]]:
        """Extract consecutive tool-call blocks from messages."""
        sequences: list[list[dict]] = []
        current: list[dict] = []

        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    current.append({
                        "tool": tc.get("function", {}).get("name", "?"),
                        "args_keys": list(
                            json.loads(tc.get("function", {}).get("arguments", "{}")).keys()
                        ) if tc.get("function", {}).get("arguments") else [],
                    })
            elif msg.get("role") in ("user", "system") and current:
                if len(current) >= self.MIN_SEQUENCE_LENGTH:
                    sequences.append(current)
                current = []
            # Keep accumulating across tool results
            elif msg.get("role") == "tool" and current:
                continue

        if current and len(current) >= self.MIN_SEQUENCE_LENGTH:
            sequences.append(current)

        return sequences

    def _sequence_signature(self, seq: list[dict]) -> str:
        """Create a stable hash of a tool call sequence."""
        parts = [s.get("tool", "?") for s in seq]
        return "→".join(parts)

    def _suggestion_from_sequence(
        self,
        sig: str,
        occurrences: list[list[dict]],
        all_tools: list[str],
    ) -> ImprovementSuggestion | None:
        """Create a suggestion from a recurring tool sequence."""
        if not occurrences:
            return None

        # Use the first occurrence as the example
        example = occurrences[0]
        tools_in_seq = list(dict.fromkeys(s.get("tool", "") for s in example))

        # Generate a descriptive name
        name_parts = sig.replace("→", "-").lower()
        name = self._slugify(f"workflow-{name_parts}")

        # Generate steps from the pattern
        steps = []
        for i, step in enumerate(example):
            tool = step.get("tool", "?")
            desc = f"Use `{tool}` to "
            if tool == "read_file":
                desc += "read relevant source files"
            elif tool == "search_files":
                desc += "search for relevant code patterns"
            elif tool == "terminal":
                desc += "run necessary commands"
            elif tool == "web_search":
                desc += "search for information online"
            else:
                desc += f"perform {tool}"
            steps.append(f"{i + 1}. {desc}")

        # Confidence based on occurrence count
        confidence = min(0.5 + len(occurrences) * 0.1, 1.0)

        return ImprovementSuggestion(
            type="new_skill",
            title=f"Recurring workflow: {sig}",
            description=f"Detected {len(occurrences)} occurrences of: {sig}. "
                        f"This pattern is worth capturing as a reusable skill.",
            pattern=example,
            tools_used=tools_in_seq,
            confidence=confidence,
            suggested_skill_name=name,
            suggested_steps=steps,
            source="pattern_detection",
        )

    def _extract_tools_from_messages(self, messages: list[dict]) -> list[str]:
        """Extract unique tool names from conversation messages."""
        tools: set[str] = set()
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    name = tc.get("function", {}).get("name", "")
                    if name:
                        tools.add(name)
        return sorted(tools)

    def _deduplicate_suggestions(
        self, suggestions: list[ImprovementSuggestion]
    ) -> list[ImprovementSuggestion]:
        """Remove duplicate suggestions by title similarity."""
        seen: set[str] = set()
        unique: list[ImprovementSuggestion] = []
        for s in suggestions:
            key = s.title.lower()[:60]
            if key not in seen:
                seen.add(key)
                unique.append(s)
        return unique

    # ── Skill rendering ─────────────────────────────────────────

    def _render_skill_md(self, suggestion: ImprovementSuggestion, name: str) -> str:
        """Render a SKILL.md from a suggestion."""
        return self.SKILL_TEMPLATE.format(
            name=name,
            description=suggestion.description[:200],
            created=time.strftime("%Y-%m-%d"),
            tools=", ".join(suggestion.tools_used),
            title=suggestion.title,
            when_to_use=f"When the agent needs to: {suggestion.description[:120]}",
            steps="\n".join(suggestion.suggested_steps) if suggestion.suggested_steps
                   else "1. Follow the pattern shown in the examples above",
            confidence=suggestion.confidence,
        )

    def _validate_skill_content(self, content: str) -> list[str]:
        """Validate a generated SKILL.md. Returns list of errors."""
        errors: list[str] = []

        # Must have YAML frontmatter
        if not content.startswith("---"):
            errors.append("Missing YAML frontmatter delimiter")

        # Must have a name field
        if "name:" not in content[:500]:
            errors.append("Missing 'name' field in frontmatter")

        # Must have content after frontmatter
        parts = content.split("---", 2)
        if len(parts) < 3:
            errors.append("Missing markdown body after frontmatter")

        # Must be under 50KB
        if len(content) > 50 * 1024:
            errors.append(f"Skill too large: {len(content)} bytes (max 50KB)")

        return errors

    def _merge_feedback(
        self,
        current: str,
        feedback: str,
        corrected_steps: list[str] | None = None,
    ) -> str:
        """Merge feedback into existing skill content."""
        lines = current.split("\n")
        new_lines = []

        in_frontmatter = False
        frontmatter_done = False

        for line in lines:
            if line.strip() == "---":
                if not in_frontmatter:
                    in_frontmatter = True
                    new_lines.append(line)
                    continue
                elif in_frontmatter and not frontmatter_done:
                    frontmatter_done = True

            if in_frontmatter and not frontmatter_done:
                # Update version in frontmatter
                if line.startswith("version:"):
                    try:
                        ver = float(line.split(":")[1].strip())
                        line = f"version: {ver + 0.1:.1f}"
                    except Exception:
                        pass
                new_lines.append(line)
            else:
                new_lines.append(line)

        # Append feedback and corrected steps
        new_lines.append("")
        new_lines.append("## Recent Improvement")
        new_lines.append(f"> {feedback}")
        if corrected_steps:
            new_lines.append("")
            new_lines.append("### Corrected Steps")
            for i, step in enumerate(corrected_steps):
                new_lines.append(f"{i + 1}. {step}")

        return "\n".join(new_lines)

    # ── Helpers ──────────────────────────────────────────────────

    def _generate_skill_name(self, suggestion: ImprovementSuggestion) -> str:
        """Generate a unique skill name from a suggestion."""
        base = suggestion.suggested_skill_name or self._slugify(suggestion.title)
        # Ensure uniqueness
        if self._manager:
            existing = set()
            if hasattr(self._manager, "list_skills"):
                for entry in self._manager.list_skills():
                    existing.add(getattr(entry, "name", ""))
            if base in existing:
                base = f"{base}-v2"
        return base

    def _write_skill_file(self, name: str, content: str) -> None:
        """Write a skill file to the skills directory."""
        if self._skills_dir:
            skill_dir = self._skills_dir
        elif self._manager and hasattr(self._manager, "_skills_dir"):
            skill_dir = Path(self._manager._skills_dir)
        else:
            skill_dir = Path.home() / ".kairos" / "skills"

        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_path = skill_dir / f"{name}.md"
        skill_path.write_text(content, encoding="utf-8")

    def _get_skill_content(self, name: str) -> str | None:
        """Read existing skill content."""
        if self._manager and hasattr(self._manager, "get_skill_content"):
            result = self._manager.get_skill_content(name)
            if result and isinstance(result, dict):
                return result.get("content", "")
        return None

    @staticmethod
    def _slugify(text: str) -> str:
        """Convert text to a safe skill name."""
        slug = re.sub(r"[^\w\s-]", "", text.lower())
        slug = re.sub(r"[-\s]+", "-", slug)
        return slug.strip("-")[:50]
