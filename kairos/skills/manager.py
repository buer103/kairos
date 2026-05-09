"""Skill Manager — lifecycle management with Curator pattern.

Skills are SKILL.md files (YAML frontmatter + Markdown body).
Curator manages lifecycle: active → stale → archived with auto-backup.
"""

from __future__ import annotations

import re
import shutil
import time
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

# Re-use the Skill dataclass from middleware
from kairos.middleware.skill_loader import Skill, _FRONTMATTER_RE


class SkillStatus(str, Enum):
    ACTIVE = "active"
    STALE = "stale"
    ARCHIVED = "archived"


class SkillEntry:
    """A skill entry with metadata and status."""

    def __init__(
        self,
        name: str,
        description: str,
        path: Path,
        status: SkillStatus = SkillStatus.ACTIVE,
        created_at: float | None = None,
        updated_at: float | None = None,
        last_used_at: float | None = None,
        use_count: int = 0,
    ):
        self.name = name
        self.description = description
        self.path = path
        self.status = status
        self.created_at = created_at or time.time()
        self.updated_at = updated_at or time.time()
        self.last_used_at = last_used_at
        self.use_count = use_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "path": str(self.path),
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_used_at": self.last_used_at,
            "use_count": self.use_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillEntry:
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            path=Path(data["path"]),
            status=SkillStatus(data.get("status", "active")),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            last_used_at=data.get("last_used_at"),
            use_count=data.get("use_count", 0),
        )


class SkillManager:
    """Curator-style skill lifecycle manager.

    Operations:
      - create: write a new SKILL.md to the skills directory
      - update: patch an existing SKILL.md (with auto-backup)
      - delete: archive a skill (with name-based forwarding for consumers)
      - list: enumerate skills with status and stats
      - stale detection: flag skills unused for >30 days

    All destructive operations create automatic backups.
    """

    BACKUP_DIR_NAME = ".backups"
    STALE_DAYS = 30

    def __init__(self, skills_dir: str | Path | None = None):
        self._dir = Path(skills_dir or Path.home() / ".kairos" / "skills")
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._dir / ".curator_index.json"
        self._backup_dir = self._dir / self.BACKUP_DIR_NAME
        self._backup_dir.mkdir(exist_ok=True)
        self._index: dict[str, SkillEntry] = {}
        self._load_index()

    # ── Public API ──────────────────────────────────────────────

    def create(self, name: str, content: str, description: str = "", category: str = "") -> Path:
        """Create a new skill. Returns the path to the SKILL.md file."""
        skill_dir = self._dir / category / name if category else self._dir / name
        skill_dir.mkdir(parents=True, exist_ok=True)

        # Build SKILL.md with YAML frontmatter
        skill_md = f"---\nname: {name}\n"
        if description:
            skill_md += f"description: {description}\n"
        skill_md += f"---\n\n{content}"

        skill_path = skill_dir / "SKILL.md"
        skill_path.write_text(skill_md, encoding="utf-8")

        entry = SkillEntry(
            name=name,
            description=description,
            path=skill_path,
            status=SkillStatus.ACTIVE,
        )
        self._index[name] = entry
        self._save_index()
        return skill_path

    def update(self, name: str, content: str | None = None, description: str | None = None) -> Path | None:
        """Update an existing skill. Auto-backs up before modifying. Returns the path."""
        entry = self._index.get(name)
        if not entry or not entry.path.exists():
            return None

        self._backup(entry)

        current = entry.path.read_text(encoding="utf-8")
        if content is not None:
            match = _FRONTMATTER_RE.match(current)
            if match:
                current = f"---\nname: {name}\n"
                if description is not None or entry.description:
                    current += f"description: {description or entry.description}\n"
                current += f"---\n\n{content}"
            else:
                current = content

        entry.path.write_text(current, encoding="utf-8")
        entry.updated_at = time.time()
        if description is not None:
            entry.description = description
        self._index[name] = entry
        self._save_index()
        return entry.path

    def delete(self, name: str, absorbed_into: str | None = None) -> bool:
        """Archive a skill. If absorbed_into is set, records the forwarding target.

        Archives the SKILL.md to .backups/ with a timestamp.
        Removes from active index.
        """
        entry = self._index.get(name)
        if not entry:
            return False

        self._backup(entry)

        # Move skill dir to backup dir with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{name}_{timestamp}"
        if absorbed_into:
            backup_name += f"_→_{absorbed_into}"

        dest = self._backup_dir / backup_name
        if entry.path.parent != self._dir:
            shutil.move(str(entry.path.parent), str(dest))
        else:
            dest.mkdir(exist_ok=True)
            shutil.move(str(entry.path), str(dest / "SKILL.md"))

        entry.status = SkillStatus.ARCHIVED
        entry.path = dest / "SKILL.md" if dest.is_dir() else dest
        entry.updated_at = time.time()
        self._index[name] = entry
        self._save_index()
        return True

    def mark_used(self, name: str) -> bool:
        """Record that a skill was used (updates last_used_at and use_count)."""
        entry = self._index.get(name)
        if not entry:
            return False
        entry.last_used_at = time.time()
        entry.use_count += 1
        if entry.status == SkillStatus.STALE:
            entry.status = SkillStatus.ACTIVE
        self._index[name] = entry
        self._save_index()
        return True

    def mark_stale(self, days: int | None = None) -> list[str]:
        """Flag skills unused for >N days as stale. Returns list of stale skill names."""
        threshold = days or self.STALE_DAYS
        cutoff = time.time() - threshold * 86400
        stale = []
        for name, entry in self._index.items():
            if entry.status != SkillStatus.ACTIVE:
                continue
            if entry.last_used_at is not None and entry.last_used_at < cutoff:
                entry.status = SkillStatus.STALE
                self._index[name] = entry
                stale.append(name)
        if stale:
            self._save_index()
        return stale

    def get(self, name: str) -> SkillEntry | None:
        """Get a skill entry by name."""
        return self._index.get(name)

    def list_skills(self, status: SkillStatus | None = None) -> list[SkillEntry]:
        """List all skills, optionally filtered by status."""
        entries = list(self._index.values())
        if status:
            entries = [e for e in entries if e.status == status]
        return sorted(entries, key=lambda e: e.name)

    def list_categories(self) -> list[str]:
        """List unique top-level categories from skill paths."""
        categories = set()
        for entry in self._index.values():
            if entry.status == SkillStatus.ARCHIVED:
                continue
            try:
                rel = entry.path.parent.relative_to(self._dir)
                if str(rel) != ".":
                    # Take the first path component as the category
                    cat = str(rel).split("/")[0]
                    if cat:
                        categories.add(cat)
            except ValueError:
                pass
        return sorted(categories)

    def stats(self) -> dict[str, Any]:
        """Return curator statistics."""
        active = sum(1 for e in self._index.values() if e.status == SkillStatus.ACTIVE)
        stale = sum(1 for e in self._index.values() if e.status == SkillStatus.STALE)
        archived = sum(1 for e in self._index.values() if e.status == SkillStatus.ARCHIVED)
        return {
            "total": len(self._index),
            "active": active,
            "stale": stale,
            "archived": archived,
            "categories": len(self.list_categories()),
            "backups": len(list(self._backup_dir.iterdir())),
        }

    def load_skill_content(self, name: str) -> Skill | None:
        """Load a skill's full content as a Skill object."""
        entry = self._index.get(name)
        if not entry or not entry.path.exists():
            return None

        content = entry.path.read_text(encoding="utf-8")
        match = _FRONTMATTER_RE.match(content)
        if not match:
            return Skill(
                name=name,
                description=entry.description,
                content=content.strip(),
                path=entry.path,
            )

        body = content[match.end():].strip()
        return Skill(
            name=name,
            description=entry.description,
            content=body,
            path=entry.path,
        )

    def resolve_forwarding(self, name: str) -> str | None:
        """If a skill was archived with a forwarding target, resolve it."""
        entry = self._index.get(name)
        if not entry or entry.status != SkillStatus.ARCHIVED:
            return None

        # Parse forwarding from backup directory name
        for d in self._backup_dir.iterdir():
            if d.name.startswith(f"{name}_") and "_→_" in d.name:
                target = d.name.split("_→_")[-1]
                return target
        return None

    # ── Internal ─────────────────────────────────────────────────

    def _load_index(self) -> None:
        if self._index_path.exists():
            try:
                import json
                data = json.loads(self._index_path.read_text())
                self._index = {
                    name: SkillEntry.from_dict(d) for name, d in data.items()
                }
            except Exception:
                self._index = {}

    def _save_index(self) -> None:
        import json
        data = {name: entry.to_dict() for name, entry in self._index.items()}
        self._index_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def _backup(self, entry: SkillEntry) -> None:
        """Create a backup of the skill file before modification."""
        if not entry.path.exists():
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self._backup_dir / f"{entry.name}_{timestamp}.bak"
        shutil.copy2(str(entry.path), str(backup_path))
