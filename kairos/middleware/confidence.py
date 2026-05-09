"""Confidence scorer — multi-dimensional output confidence with evidence citation.

Kairos original middleware. Scores every agent response using:
  1. Completion ratio (completed / total steps)
  2. Tool diversity (distinct tools used)
  3. Error-free ratio (tool outputs without errors)
  4. Citation density (references to sources/evidence)
  5. Response specificity (vague vs actionable language)
  6. Recency weighting (recent steps weighted higher)

Confidence levels trigger actions:
  HIGH (≥0.8): Direct answer
  MEDIUM (0.5-0.8): Answer with disclaimer
  LOW (0.3-0.5): Request user verification
  CRITICAL (<0.3): Auto-invoke clarification
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from kairos.core.middleware import Middleware

logger = logging.getLogger("kairos.middleware.confidence")


# ── Confidence Level ────────────────────────────────────────────

@dataclass
class ConfidenceResult:
    """Structured confidence score with breakdown and evidence citations."""
    overall: float
    level: str  # HIGH / MEDIUM / LOW / CRITICAL
    breakdown: dict[str, float] = field(default_factory=dict)
    citations: list[str] = field(default_factory=list)
    recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "confidence": self.overall,
            "level": self.level,
            "breakdown": self.breakdown,
            "citations": self.citations,
            "recommendation": self.recommendation,
        }


# ── Confidence Scorer ───────────────────────────────────────────

class ConfidenceScorer(Middleware):
    """Evaluates output confidence using multi-factor scoring.

    Hook: after_agent — scores the completed session and sets case.confidence.

    Usage:
        scorer = ConfidenceScorer(
            completion_weight=0.25,
            error_weight=0.25,
            diversity_weight=0.15,
            specificity_weight=0.15,
            citation_weight=0.10,
            recency_weight=0.10,
            low_threshold=0.3,
            med_threshold=0.5,
            high_threshold=0.8,
        )
    """

    def __init__(
        self,
        completion_weight: float = 0.25,
        error_weight: float = 0.25,
        diversity_weight: float = 0.15,
        specificity_weight: float = 0.15,
        citation_weight: float = 0.10,
        recency_weight: float = 0.10,
        low_threshold: float = 0.3,
        med_threshold: float = 0.5,
        high_threshold: float = 0.8,
        nlp_scoring: bool = False,
        nlp_model: Any = None,
    ):
        self._w_completion = completion_weight
        self._w_error = error_weight
        self._w_diversity = diversity_weight
        self._w_specificity = specificity_weight
        self._w_citation = citation_weight
        self._w_recency = recency_weight
        self._low = low_threshold
        self._med = med_threshold
        self._high = high_threshold
        self._nlp_scoring = nlp_scoring
        self._nlp_model = nlp_model
        self._history: list[ConfidenceResult] = []

    # ── after_agent — main scoring ──────────────────────────────

    def after_agent(self, state: Any, runtime: dict[str, Any]) -> dict[str, Any] | None:
        if not state.case or not state.case.steps:
            return None

        steps = state.case.steps
        completed = [s for s in steps if s.result is not None]
        if not completed:
            state.case.confidence = 0.1
            self._record(ConfidenceResult(
                overall=0.1, level="CRITICAL",
                recommendation="No completed steps — agent produced no tool-based evidence.",
            ))
            return None

        # Extract final response
        final_content = self._get_final_content(state)

        # ── Factor 1: Completion ratio ─────────────────────────
        f_completion = self._score_completion(steps, completed)

        # ── Factor 2: Error-free ratio ─────────────────────────
        f_error = self._score_error_free(completed)

        # ── Factor 3: Tool diversity ───────────────────────────
        f_diversity = self._score_diversity(completed)

        # ── Factor 4: Response specificity ─────────────────────
        f_specificity = self._score_specificity(final_content)

        # ── Factor 5: Citation density ─────────────────────────
        f_citation = self._score_citation(final_content)

        # ── Factor 6: Recency weighting ────────────────────────
        f_recency = self._score_recency(completed)

        # ── Weighted combination ───────────────────────────────
        overall = (
            f_completion * self._w_completion
            + f_error * self._w_error
            + f_diversity * self._w_diversity
            + f_specificity * self._w_specificity
            + f_citation * self._w_citation
            + f_recency * self._w_recency
        )
        overall = round(min(overall, 0.99), 4)

        # ── Confidence level ───────────────────────────────────
        if overall >= self._high:
            level = "HIGH"
            recommendation = "Direct answer — high confidence."
        elif overall >= self._med:
            level = "MEDIUM"
            recommendation = "Answer with disclaimer — moderate confidence."
        elif overall >= self._low:
            level = "LOW"
            recommendation = "Request user verification — low confidence."
        else:
            level = "CRITICAL"
            recommendation = "Auto-invoke clarification — insufficient evidence."

        # ── Build citations ────────────────────────────────────
        citations = self._build_citations(completed)

        # ── Store result ───────────────────────────────────────
        result = ConfidenceResult(
            overall=overall,
            level=level,
            breakdown={
                "completion": round(f_completion, 3),
                "error_free": round(f_error, 3),
                "diversity": round(f_diversity, 3),
                "specificity": round(f_specificity, 3),
                "citation": round(f_citation, 3),
                "recency": round(f_recency, 3),
            },
            citations=citations,
            recommendation=recommendation,
        )

        state.case.confidence = overall
        state.case.confidence_result = result.to_dict()

        if hasattr(state.case, "evidence_summary"):
            state.case.evidence_summary = [
                {
                    "step": s.id,
                    "tool": s.tool,
                    "finding": str(s.result)[:200] if s.result else "no result",
                    "duration_ms": s.duration_ms,
                }
                for s in completed
            ]

        self._record(result)
        return None

    # ── Scoring factors ────────────────────────────────────────

    @staticmethod
    def _score_completion(steps: list, completed: list) -> float:
        """Completion ratio: 0.3–1.0."""
        ratio = len(completed) / max(len(steps), 1)
        return 0.3 + ratio * 0.7

    @staticmethod
    def _score_error_free(completed: list) -> float:
        """Error-free ratio among completed steps."""
        error_free = [
            s for s in completed
            if not ConfidenceScorer._step_has_error(s)
        ]
        return len(error_free) / max(len(completed), 1)

    @staticmethod
    def _score_diversity(completed: list) -> float:
        """Tool diversity: more distinct tools → higher confidence."""
        distinct = len(set(s.tool for s in completed))
        return min(distinct / 3, 1.0)  # Cap at 3 distinct tools

    @staticmethod
    def _score_specificity(content: str) -> float:
        """Response specificity: actionable vs vague language."""
        if not content:
            return 0.1

        # Vague patterns reduce confidence
        vague = len(re.findall(
            r"\b(maybe|perhaps|possibly|could be|might be|I think|not sure|unclear)\b",
            content, re.IGNORECASE,
        ))
        # Specific patterns increase confidence
        specific = len(re.findall(
            r"\b(specifically|exactly|confirmed|verified|according to|based on|"
            r"the result shows|evidence indicates|step \d|file |path |error |"
            r"successfully|completed|returned)\b",
            content, re.IGNORECASE,
        ))
        # Code blocks or structured output
        structured = content.count("```") // 2

        base = 0.5
        base -= min(vague * 0.15, 0.4)
        base += min(specific * 0.08, 0.3)
        base += min(structured * 0.05, 0.15)
        return max(0.1, min(base, 1.0))

    @staticmethod
    def _score_citation(content: str) -> float:
        """Citation density: references to sources/evidence/tools."""
        if not content:
            return 0.0

        patterns = [
            r"\[.*?\]\(.*?\)",           # Markdown links
            r"`[^`]+`",                   # Inline code (tool names, paths)
            r"(step|tool|evidence|source)[\s:-]*\d",  # Step/tool references
            r"https?://\S+",             # URLs
            r"(according to|based on|from|in) `",  # Contextual citations
        ]
        count = sum(
            len(re.findall(p, content, re.IGNORECASE))
            for p in patterns
        )
        # Scale: 0 citations → 0.0, 10+ → 1.0
        return min(count / 10, 1.0)

    @staticmethod
    def _score_recency(completed: list) -> float:
        """Recency: later steps weighted higher than early ones."""
        if not completed:
            return 0.0
        n = len(completed)
        total_weight = sum(i + 1 for i in range(n))  # 1 + 2 + ... + n
        weighted = sum(
            (i + 1) * (1.0 if not ConfidenceScorer._step_has_error(s) else 0.5)
            for i, s in enumerate(completed)
        )
        return weighted / max(total_weight, 1)

    # ── Helpers ────────────────────────────────────────────────

    @staticmethod
    def _step_has_error(step: Any) -> bool:
        """Check if a step's result contains an error."""
        result = step.result if hasattr(step, "result") else None
        if result is None:
            return False
        if isinstance(result, dict):
            return "error" in result
        if isinstance(result, str):
            return bool(re.search(r"(?i)(error|exception|failed|traceback)", result))
        return False

    @staticmethod
    def _get_final_content(state: Any) -> str:
        """Extract the final assistant message content."""
        if hasattr(state, "messages"):
            for msg in reversed(state.messages):
                if msg.get("role") == "assistant" and msg.get("content"):
                    return msg["content"]
        return ""

    @staticmethod
    def _build_citations(completed: list) -> list[str]:
        """Build citation strings from completed steps."""
        citations = []
        for s in completed:
            tool = s.tool if hasattr(s, "tool") else "?"
            result = s.result if hasattr(s, "result") else None
            snippet = ""
            if isinstance(result, dict):
                snippet = str(result.get("content", result.get("result", "")))[:80]
            elif isinstance(result, str):
                snippet = result[:80]
            citations.append(f"[{tool}]: {snippet}..." if snippet else f"[{tool}]")
        return citations

    def _record(self, result: ConfidenceResult) -> None:
        """Record confidence result for observability."""
        self._history.append(result)
        logger.debug(
            "Confidence: %.3f (%s) — %s",
            result.overall, result.level, result.recommendation,
        )

    @property
    def history(self) -> list[ConfidenceResult]:
        return self._history

    def __repr__(self) -> str:
        return (
            f"ConfidenceScorer(low={self._low}, med={self._med}, "
            f"high={self._high}, nlp={self._nlp_scoring})"
        )
