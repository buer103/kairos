"""Skill loader middleware — loads and injects skills into the agent context."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from kairos.core.middleware import Middleware


# YAML frontmatter + Markdown pattern
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


class Skill:
    """A loaded skill with metadata and content."""

    def __init__(self, name: str, description: str, content: str, path: Path):
        self.name = name
        self.description = description
        self.content = content
        self.path = path

    def __repr__(self) -> str:
        return f"Skill({self.name!r})"


class SkillLoader(Middleware):
    """Loads SKILL.md files and injects relevant skills into context.

    Hook: before_agent — scans and loads all skills.
          before_model — injects matching skill instructions.

    Skills live in SKILL.md files with YAML frontmatter:

        ---
        name: my-skill
        description: What this skill does
        ---
        # Skill content here...
    """

    def __init__(self, skills_dir: str | Path | None = None):
        self._skills_dir = Path(skills_dir or Path.home() / ".kairos" / "skills")
        self._skills: list[Skill] = []
        self._loaded = False

    def before_agent(self, state: Any, runtime: dict[str, Any]) -> dict[str, Any] | None:
        """Scan and load all SKILL.md files from the skills directory."""
        if self._loaded:
            return None

        self._skills = []
        if not self._skills_dir.exists():
            return None

        for skill_file in self._skills_dir.glob("**/SKILL.md"):
            try:
                content = skill_file.read_text(encoding="utf-8")
                skill = self._parse_skill(content, skill_file)
                if skill:
                    self._skills.append(skill)
            except Exception:
                pass  # Skip unreadable files

        self._loaded = True
        return {"skills_loaded": len(self._skills)}

    def before_model(self, state: Any, runtime: dict[str, Any]) -> dict[str, Any] | None:
        """Inject relevant skill descriptions into the system prompt.

        Currently injects a list of available skills. In future: semantic matching
        to select only relevant skills based on the current conversation.
        """
        if not self._skills:
            return None

        # Check if we've already injected skills
        if runtime.get("_skills_injected"):
            return None

        skill_list = "\n".join(
            f"- **{s.name}**: {s.description}" for s in self._skills
        )
        skill_block = f"""\n\n## Available Skills

The following specialized skills are available. When relevant, describe which
skill you would use and why.

{skill_list}
"""

        # Inject into system message
        if state.messages and state.messages[0].get("role") == "system":
            state.messages[0]["content"] += skill_block
            runtime["_skills_injected"] = True

        return None

    @staticmethod
    def _parse_skill(content: str, path: Path) -> Skill | None:
        """Parse YAML frontmatter + Markdown body from a SKILL.md file."""
        match = _FRONTMATTER_RE.match(content)
        if not match:
            name = path.parent.name
            return Skill(
                name=name,
                description="",
                content=content.strip(),
                path=path,
            )

        frontmatter = match.group(1)
        body = content[match.end():].strip()

        # Simple YAML extraction (no pyyaml dependency)
        name = ""
        description = ""
        for line in frontmatter.split("\n"):
            line = line.strip()
            if line.startswith("name:"):
                name = line.split(":", 1)[1].strip().strip('"').strip("'")
            elif line.startswith("description:"):
                description = line.split(":", 1)[1].strip().strip('"').strip("'")

        if not name:
            name = path.parent.name

        return Skill(name=name, description=description, content=body, path=path)

    def get_skills(self) -> list[Skill]:
        """Get all loaded skills."""
        return list(self._skills)

    def find_skill(self, name: str) -> Skill | None:
        """Find a skill by name."""
        for s in self._skills:
            if s.name == name:
                return s
        return None
