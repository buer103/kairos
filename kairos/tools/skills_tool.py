"""Built-in skill management tools — skills_list, skill_view, skill_manage.

These tools allow the AI agent to discover, inspect, and manage skills at
runtime, mirroring Hermes' skill system.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kairos.tools.registry import register_tool

# Module-level manager — injected by Agent at startup
_skill_manager: Any = None


def set_skill_manager(manager: Any) -> None:
    """Inject a SkillManager instance for skill tools."""
    global _skill_manager
    _skill_manager = manager


def get_skill_manager() -> Any:
    """Get the current SkillManager instance."""
    return _skill_manager


@register_tool(
    name="skills_list",
    description="List available skills with their names, descriptions, and status. "
    "Use this to discover what skills are available before loading one with skill_view.",
    parameters={
        "status": {
            "type": "string",
            "description": "Filter by status: 'active', 'stale', or omit for all non-archived",
        },
        "category": {
            "type": "string",
            "description": "Filter by category (e.g., 'creative', 'devops')",
        },
    },
)
def skills_list(status: str = "", category: str = "") -> dict[str, Any]:
    """List available skills."""
    if _skill_manager is None:
        return {"skills": [], "error": "Skill manager not initialized"}

    from kairos.skills.manager import SkillStatus

    status_filter = None
    if status:
        try:
            status_filter = SkillStatus(status)
        except ValueError:
            return {"skills": [], "error": f"Invalid status: {status}"}

    entries = _skill_manager.list_skills(status=status_filter)

    # Format for AI consumption
    skills = []
    for entry in entries:
        if entry.status == SkillStatus.ARCHIVED:
            continue

        skill_data = {
            "name": entry.name,
            "description": entry.description,
            "status": entry.status.value,
            "use_count": entry.use_count,
            "category": "",
        }

        # Extract category from path
        try:
            rel = entry.path.parent.relative_to(_skill_manager._dir)
            parts = str(rel).split("/")
            if len(parts) > 1:
                skill_data["category"] = parts[0]
        except Exception:
            pass

        if category and skill_data["category"] != category:
            continue

        skills.append(skill_data)

    return {"skills": skills, "total": len(skills)}


@register_tool(
    name="skill_view",
    description="Load a skill's full content including its instructions, metadata, and "
    "list of linked files (references, templates, scripts). Use this before executing "
    "a task the skill covers.",
    parameters={
        "name": {
            "type": "string",
            "description": "Name of the skill to load (e.g., 'code-review', 'agent-framework-design')",
        },
        "file_path": {
            "type": "string",
            "description": "Optional: path to a specific file within the skill "
            "(e.g., 'references/api.md', 'scripts/setup.py')",
        },
    },
)
def skill_view(name: str, file_path: str = "") -> dict[str, Any]:
    """Load a skill's full content."""
    if _skill_manager is None:
        return {"success": False, "error": "Skill manager not initialized"}

    # If requesting a specific file
    if file_path:
        result = _skill_manager.list_files(name, file_path)
        if result is None:
            return {"success": False, "error": f"Skill not found: {name}"}
        if "error" in result:
            return {"success": False, "error": result["error"]}
        return {"success": True, **result}

    # Load full skill
    content = _skill_manager.get_skill_content(name)
    if content is None:
        # Try marketplace
        if hasattr(_skill_manager, "_index"):
            entry = _skill_manager._index.get(name)
            if entry:
                return {
                    "success": False,
                    "error": f"Skill '{name}' is archived. "
                    f"Use skill_manage to restore or un-archive.",
                }
        return {
            "success": False,
            "error": f"Skill not found: {name}. "
            f"Use skills_list to see available skills.",
        }

    # Record usage
    _skill_manager.mark_used(name)

    # Format response
    setup_needed = content.get("metadata", {}).get("requires", "")
    setup_note = ""
    if setup_needed:
        setup_note = f"Setup required: {setup_needed}"

    return {
        "success": True,
        "name": content["name"],
        "description": content["description"],
        "content": content["content"],
        "skill_dir": content["skill_dir"],
        "status": content["status"],
        "use_count": content["use_count"],
        "metadata": content["metadata"],
        "linked_files": {
            k: v for k, v in content.get("linked_files", {}).items() if v
        },
        "setup_note": setup_note,
    }


@register_tool(
    name="skill_manage",
    description="Create, update, or delete skills. "
    "Use 'create' to write a new skill, 'update' to modify existing, "
    "'delete' to archive a skill. For delete, set absorbed_into to "
    "forward consumers to a replacement skill.",
    parameters={
        "action": {
            "type": "string",
            "description": "One of: create, update, delete",
        },
        "name": {
            "type": "string",
            "description": "Name of the skill (lowercase, hyphens, max 64 chars)",
        },
        "content": {
            "type": "string",
            "description": "Skill markdown content (required for create/update). "
            "For create, include a YAML frontmatter block with name and description.",
        },
        "description": {
            "type": "string",
            "description": "Short description of the skill",
        },
        "category": {
            "type": "string",
            "description": "Optional category folder for organizing skills",
        },
        "absorbed_into": {
            "type": "string",
            "description": "For delete: name of the skill that replaces this one",
        },
    },
)
def skill_manage(
    action: str,
    name: str,
    content: str = "",
    description: str = "",
    category: str = "",
    absorbed_into: str = "",
) -> dict[str, Any]:
    """Manage skills — create, update, delete."""
    if _skill_manager is None:
        return {"success": False, "error": "Skill manager not initialized"}

    action = action.lower().strip()

    if action == "create":
        if not content:
            return {"success": False, "error": "Content is required for create"}
        path = _skill_manager.create(
            name=name,
            content=content,
            description=description,
            category=category,
        )
        return {
            "success": True,
            "name": name,
            "path": str(path),
            "message": f"Skill '{name}' created at {path}",
        }

    elif action == "update":
        if not content:
            return {"success": False, "error": "Content is required for update"}
        path = _skill_manager.update(
            name=name,
            content=content,
            description=description or None,
        )
        if path is None:
            return {"success": False, "error": f"Skill not found: {name}"}
        return {
            "success": True,
            "name": name,
            "path": str(path),
            "message": f"Skill '{name}' updated (backup created)",
        }

    elif action == "delete":
        absorbed = absorbed_into or None
        ok = _skill_manager.delete(name, absorbed_into=absorbed)
        if not ok:
            return {"success": False, "error": f"Skill not found: {name}"}
        msg = f"Skill '{name}' archived"
        if absorbed:
            msg += f" (consumers forwarded to '{absorbed}')"
        return {"success": True, "name": name, "message": msg}

    else:
        return {"success": False, "error": f"Unknown action: {action}. Use create/update/delete."}
