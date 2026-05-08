"""Evidence chain database — persistent storage for reasoning traces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kairos.core.state import Case, Step


class EvidenceDB:
    """
    Persistent storage for evidence chains.

    Each case is stored as a JSON file under ~/.kairos/evidence/.
    """

    def __init__(self, base_path: str | Path | None = None):
        self._base = Path(base_path or Path.home() / ".kairos" / "evidence")
        self._base.mkdir(parents=True, exist_ok=True)

    def save(self, case: Case) -> Path:
        """Persist a case to disk."""
        data = {
            "id": case.id,
            "created_at": case.created_at.isoformat(),
            "conclusion": case.conclusion,
            "confidence": case.confidence,
            "steps": [
                {
                    "id": s.id,
                    "tool": s.tool,
                    "args": s.args,
                    "result": s.result,
                    "duration_ms": s.duration_ms,
                }
                for s in case.steps
            ],
        }
        path = self._base / f"{case.id}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        return path

    def load(self, case_id: str) -> Case | None:
        """Load a case from disk."""
        path = self._base / f"{case_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        case = Case(id=data["id"])
        case.conclusion = data.get("conclusion")
        case.confidence = data.get("confidence")
        for s_data in data.get("steps", []):
            step = Step(
                id=s_data["id"],
                tool=s_data["tool"],
                args=s_data["args"],
                result=s_data.get("result"),
                duration_ms=s_data.get("duration_ms", 0.0),
            )
            case.steps.append(step)
        return case

    def list_cases(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recent cases."""
        files = sorted(self._base.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        results = []
        for f in files[:limit]:
            data = json.loads(f.read_text())
            results.append({
                "id": data["id"],
                "created_at": data.get("created_at"),
                "conclusion": data.get("conclusion"),
                "confidence": data.get("confidence"),
                "steps_count": len(data.get("steps", [])),
            })
        return results
