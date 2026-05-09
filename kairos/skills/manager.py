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
        self._external_dirs: list[Path] = []
        self._load_index()
        self._load_config()
        self._ensure_builtin()

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

    # ── Scanning & Reindex ──────────────────────────────────────

    def scan(self, categories: list[str] | None = None) -> dict[str, Any]:
        """Scan the skills directory for SKILL.md files and update the index.

        Discovers new skills not yet in the index. Does NOT remove index
        entries for files that are gone — use reindex() for that.

        Returns a dict with added, updated, unchanged counts.
        """
        if not self._dir.exists():
            return {"added": 0, "updated": 0, "unchanged": 0}

        added = 0
        updated = 0
        unchanged = 0
        seen: set[str] = set()

        search_dirs = [self._dir] + list(self._external_dirs)
        if categories:
            search_dirs = [self._dir / c for c in categories if (self._dir / c).exists()]

        for search_dir in search_dirs:
            for skill_md in search_dir.glob("**/SKILL.md"):
                # Skip backup dir
                if self.BACKUP_DIR_NAME in skill_md.parts:
                    continue

                try:
                    content = skill_md.read_text(encoding="utf-8")
                    match = _FRONTMATTER_RE.match(content)
                    if match:
                        frontmatter = match.group(1)
                        name = ""
                        description = ""
                        for line in frontmatter.split("\n"):
                            line = line.strip()
                            if line.startswith("name:"):
                                name = line.split(":", 1)[1].strip().strip('"').strip("'")
                            elif line.startswith("description:"):
                                description = line.split(":", 1)[1].strip().strip('"').strip("'")
                        if not name:
                            name = skill_md.parent.name
                    else:
                        name = skill_md.parent.name
                        description = ""

                    seen.add(name)

                    existing = self._index.get(name)
                    if existing:
                        if existing.status == SkillStatus.ARCHIVED:
                            existing.status = SkillStatus.ACTIVE
                            existing.path = skill_md
                            existing.updated_at = time.time()
                            updated += 1
                        else:
                            unchanged += 1
                    else:
                        self._index[name] = SkillEntry(
                            name=name,
                            description=description,
                            path=skill_md,
                            status=SkillStatus.ACTIVE,
                        )
                        added += 1
                except Exception:
                    pass

        if added or updated:
            self._save_index()

        return {"added": added, "updated": updated, "unchanged": unchanged}

    def reindex(self) -> dict[str, Any]:
        """Full rescan: add new skills, remove stale index entries, update changed.

        Returns a dict with added, removed, updated, unchanged counts.
        """
        if not self._dir.exists():
            removed = len(self._index)
            self._index.clear()
            self._save_index()
            return {"added": 0, "removed": removed, "updated": 0, "unchanged": 0}

        # Build set of names found on disk
        found: set[str] = set()
        for search_dir in [self._dir] + list(self._external_dirs):
            for skill_md in search_dir.glob("**/SKILL.md"):
                if self.BACKUP_DIR_NAME in skill_md.parts:
                    continue
                try:
                    content = skill_md.read_text(encoding="utf-8")
                    match = _FRONTMATTER_RE.match(content)
                    if match:
                        frontmatter = match.group(1)
                        name = ""
                        description = ""
                        for line in frontmatter.split("\n"):
                            line = line.strip()
                            if line.startswith("name:"):
                                name = line.split(":", 1)[1].strip().strip('"').strip("'")
                            elif line.startswith("description:"):
                                description = line.split(":", 1)[1].strip().strip('"').strip("'")
                        if not name:
                            name = skill_md.parent.name
                    else:
                        name = skill_md.parent.name
                        description = ""

                    found.add(name)

                    existing = self._index.get(name)
                    if existing:
                        if existing.status == SkillStatus.ARCHIVED:
                            existing.status = SkillStatus.ACTIVE
                            existing.path = skill_md
                            existing.updated_at = time.time()
                        else:
                            existing.path = skill_md  # Update path (might have moved)
                    else:
                        self._index[name] = SkillEntry(
                            name=name,
                            description=description,
                            path=skill_md,
                            status=SkillStatus.ACTIVE,
                        )
                except Exception:
                    pass

        # Remove entries not found on disk (archived ones excluded from deletion)
        removed = 0
        stale_names = []
        for name, entry in list(self._index.items()):
            if name not in found and entry.status != SkillStatus.ARCHIVED:
                stale_names.append(name)

        for name in stale_names:
            del self._index[name]
            removed += 1

        self._save_index()

        return {"added": max(0, len(found) - len(self._index) + removed),
                "removed": removed, "updated": 0, "unchanged": len(found) - max(0, len(found) - len(self._index) + removed)}

    def clean(self, days: int | None = None) -> dict[str, Any]:
        """Remove all archived skills older than N days from the backup directory.

        Returns counts of cleaned entries and freed bytes.
        """
        threshold_days = days or (self.STALE_DAYS * 3)  # Default: 90 days
        cutoff = time.time() - threshold_days * 86400
        cleaned = 0
        freed_bytes = 0

        if not self._backup_dir.exists():
            return {"cleaned": 0, "freed_bytes": 0}

        for item in self._backup_dir.iterdir():
            try:
                stat = item.stat()
                if stat.st_mtime < cutoff:
                    if item.is_dir():
                        for f in item.rglob("*"):
                            if f.is_file():
                                freed_bytes += f.stat().st_size
                        import shutil
                        shutil.rmtree(item)
                    else:
                        freed_bytes += stat.st_size
                        item.unlink()
                    cleaned += 1
            except Exception:
                pass

        # Also remove archived entries from index that are past threshold
        to_remove = []
        for name, entry in list(self._index.items()):
            if entry.status == SkillStatus.ARCHIVED:
                if entry.updated_at < cutoff:
                    to_remove.append(name)

        for name in to_remove:
            del self._index[name]

        if to_remove:
            self._save_index()

        return {"cleaned": cleaned + len(to_remove), "freed_bytes": freed_bytes}

    # ── Bulk Access ─────────────────────────────────────────────

    def get_all_skills(self) -> list[dict[str, Any]]:
        """Return all active/stale skills with full metadata for tool consumption."""
        result = []
        for entry in self._index.values():
            if entry.status == SkillStatus.ARCHIVED:
                continue
            data = entry.to_dict()
            # Add frontmatter metadata if file exists
            if entry.path.exists():
                try:
                    content = entry.path.read_text(encoding="utf-8")
                    match = _FRONTMATTER_RE.match(content)
                    if match:
                        fm = match.group(1)
                        for line in fm.split("\n"):
                            line = line.strip()
                            if ":" in line and not line.startswith("#"):
                                k, _, v = line.partition(":")
                                k = k.strip()
                                v = v.strip().strip('"').strip("'")
                                if k not in ("name", "description"):
                                    data[k] = v
                except Exception:
                    pass
            result.append(data)
        return sorted(result, key=lambda s: s["name"])

    def get_skill_content(self, name: str) -> dict[str, Any] | None:
        """Load a skill's full content with metadata, body, and linked files.

        Returns None if the skill doesn't exist or is archived.
        """
        entry = self._index.get(name)
        if not entry or entry.status == SkillStatus.ARCHIVED:
            return None

        if not entry.path.exists():
            return None

        try:
            raw = entry.path.read_text(encoding="utf-8")
        except Exception:
            return None

        match = _FRONTMATTER_RE.match(raw)
        if match:
            frontmatter = match.group(1)
            body = raw[match.end():].strip()
        else:
            frontmatter = ""
            body = raw.strip()

        # Parse frontmatter
        metadata: dict[str, str] = {}
        for line in frontmatter.split("\n"):
            line = line.strip()
            if ":" in line and not line.startswith("#"):
                k, _, v = line.partition(":")
                metadata[k.strip()] = v.strip().strip('"').strip("'")

        # Discover linked files
        linked_files: dict[str, list[str]] = {"references": [], "templates": [], "scripts": [], "assets": []}
        skill_dir = entry.path.parent
        for subdir in linked_files:
            subdir_path = skill_dir / subdir
            if subdir_path.exists():
                for f in sorted(subdir_path.rglob("*")):
                    if f.is_file():
                        linked_files[subdir].append(str(f.relative_to(skill_dir)))

        return {
            "name": entry.name,
            "description": entry.description,
            "content": body,
            "raw_content": raw,
            "path": str(entry.path),
            "skill_dir": str(skill_dir),
            "status": entry.status.value,
            "use_count": entry.use_count,
            "metadata": metadata,
            "linked_files": linked_files,
        }

    def list_files(self, name: str, file_path: str | None = None) -> dict[str, Any] | None:
        """List or access files within a skill directory.

        If file_path is None, returns list of linked files.
        If file_path is set, returns the file content.
        """
        entry = self._index.get(name)
        if not entry or entry.status == SkillStatus.ARCHIVED:
            return None

        skill_dir = entry.path.parent

        if file_path:
            # Resolve and read a specific file
            target = (skill_dir / file_path).resolve()
            try:
                target.relative_to(skill_dir.resolve())
            except ValueError:
                return {"error": "Path traversal denied"}

            if not target.exists() or not target.is_file():
                return {"error": f"File not found: {file_path}"}

            try:
                content = target.read_text(encoding="utf-8")
                return {"path": str(target), "content": content, "size": len(content)}
            except UnicodeDecodeError:
                return {"path": str(target), "binary": True, "size": target.stat().st_size}

        # List all files
        files: dict[str, list[str]] = {}
        for subdir in ("references", "templates", "scripts", "assets"):
            subdir_path = skill_dir / subdir
            if subdir_path.exists():
                files[subdir] = [str(f.relative_to(skill_dir)) for f in sorted(subdir_path.rglob("*")) if f.is_file()]

        result = self.get_skill_content(name) or {}
        result["skill_dir"] = str(skill_dir)
        result["files"] = files
        return result

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

    def _load_config(self) -> None:
        """Load skills configuration from kairos config.yaml if available."""
        try:
            from kairos.config import get_config
            cfg = get_config()
            # Load external skill directories
            ext_dirs = cfg.get("skills.external_dirs", [])
            if ext_dirs:
                for d in ext_dirs:
                    p = Path(d).expanduser()
                    if p.exists():
                        self._external_dirs.append(p)
            # Override stale days from config
            stale_days = cfg.get("skills.stale_days")
            if stale_days is not None:
                self.STALE_DAYS = int(stale_days)
        except Exception:
            pass

    def _ensure_builtin(self) -> None:
        """Copy built-in skills from the package to the user skills dir on first run."""
        # Find the package's builtin skills directory
        import kairos.skills as skills_pkg
        pkg_dir = Path(skills_pkg.__file__).parent
        builtin_src = pkg_dir / "builtin"
        builtin_dst = self._dir / "builtin"

        if not builtin_src.exists():
            return

        # Only copy if destination doesn't exist (first run)
        if builtin_dst.exists():
            return

        try:
            shutil.copytree(str(builtin_src), str(builtin_dst))
        except Exception:
            pass
