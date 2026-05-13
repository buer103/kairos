"""CLI entry point for Kairos — Rich TUI, chat, cron, and config commands.

Usage (Hermes-style):
    kairos                    Interactive chat mode (default) with live streaming
    kairos <query>            One-shot query (no 'run' needed)
    kairos chat               Explicit chat mode
    kairos run <query>        One-shot query (explicit)
    kairos --resume <name>    Resume a saved session
    kairos --list-sessions    List saved sessions
    kairos --version          Show version
    kairos cron|config|skill|curator|tui|web  Management subcommands
"""

from __future__ import annotations

import os
import sys
import time

from kairos.cli.rich_ui import KairosConsole, SKINS

__all__ = ["KairosConsole", "SKINS", "main"]


def main():
    """Main entry point."""
    args = sys.argv[1:]

    # ── Extract --base-url and --model from anywhere in args ───
    base_url: str | None = None
    model_name: str | None = None
    filtered: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--base-url" and i + 1 < len(args):
            base_url = args[i + 1]
            i += 2
        elif args[i].startswith("--base-url="):
            base_url = args[i].split("=", 1)[1]
            i += 1
        elif args[i] == "--model" and i + 1 < len(args):
            model_name = args[i + 1]
            i += 2
        elif args[i].startswith("--model="):
            model_name = args[i].split("=", 1)[1]
            i += 1
        else:
            filtered.append(args[i])
            i += 1
    args = filtered

    # ── No args → interactive chat ─────────────────────────────
    if not args:
        _chat_mode([], base_url=base_url, model_name=model_name)
        return

    # ── Flags ─────────────────────────────────────────────────
    if args[0] == "--version":
        from kairos import __version__
        print(f"kairos {__version__}")
        return

    if args[0] == "--help" or args[0] == "-h":
        _print_usage()
        return

    if args[0] == "--list-sessions":
        _list_sessions_cmd()
        return

    if args[0] == "--resume" and len(args) >= 2:
        _chat_mode([], resume=args[1], base_url=base_url, model_name=model_name)
        return

    # ── Subcommands ───────────────────────────────────────────
    if args[0] == "chat":
        _chat_mode(args[1:], base_url=base_url, model_name=model_name)
    elif args[0] == "run":
        _run_mode(args[1:], base_url=base_url, model_name=model_name)
    elif args[0] == "cron":
        _cron_mode(args[1:])
    elif args[0] == "config":
        _config_mode(args[1:])
    elif args[0] == "skill":
        _skill_mode(args[1:])
    elif args[0] == "curator":
        _curator_mode(args[1:])
    elif args[0] == "doctor":
        _doctor_mode()
    elif args[0] == "web":
        _web_mode(base_url=base_url, model_name=model_name)
    elif args[0] == "tui":
        _tui_mode(argv=sys.argv)
    elif args[0].startswith("-"):
        print(f"Unknown flag: {args[0]}")
        _print_usage()
    else:
        # ── Bare query → one-shot run (Hermes-style) ──────────
        _run_mode(args, base_url=base_url, model_name=model_name)


def _print_usage():
    msg = """Kairos — The right tool, at the right moment.

Usage:
  kairos [--base-url URL] [--model NAME]           Interactive chat mode
  kairos [--base-url URL] [--model NAME] <query>   One-shot query
  kairos chat  [--base-url URL] [--model NAME]     Explicit interactive chat
  kairos run   [--base-url URL] [--model NAME] <q> One-shot query (explicit)
  kairos --resume <name>                            Resume a saved session
  kairos --list-sessions                            List saved sessions

Provider flags:
  --base-url URL      API endpoint (env: KAIROS_BASE_URL)
  --model NAME        Model name (env: KAIROS_MODEL)

Examples:
  kairos                                                    # DeepSeek chat (default)
  kairos --base-url http://localhost:8000/v1 --model qwen   # vLLM chat
  kairos --model gpt-4o "Hello"                             # OpenAI one-shot

Management:
  kairos cron                Cron scheduler management
  kairos config init         Generate default config file
  kairos skill list          List installed skills
  kairos curator status      Show skill lifecycle status
  kairos doctor               Health check
  kairos config migrate       Upgrade old config with new defaults
  kairos tui [--skin NAME]    Launch Textual TUI (interactive)
  kairos web                  Launch Web UI (http://127.0.0.1:8080)
  kairos --version           Show version"""
    print(msg)


def _list_sessions_cmd():
    """CLI: kairos --list-sessions"""
    from kairos.core.stateful_agent import StatefulAgent
    from kairos.providers.base import ModelConfig

    api_key = _get_api_key()
    if not api_key:
        print("No API key found. Set DEEPSEEK_API_KEY or run 'kairos config init'.")
        return

    agent = StatefulAgent(model=ModelConfig(api_key=api_key))
    sessions = agent.list_sessions()
    if not sessions:
        print("No saved sessions.")
        return

    print(f"Saved sessions ({len(sessions)}):\n")
    for s in sessions:
        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(s.get("saved_at", 0)))
        msgs = s.get("message_count", 0)
        turns = s.get("turn_count", 0)
        print(f"  ● {s['name']}")
        print(f"    id={s.get('session_id','?')}  messages={msgs}  turns={turns}  saved={ts}")


def _get_model_config(base_url: str | None = None, model_name: str | None = None) -> ModelConfig | None:
    """Build ModelConfig from env vars, config file, and CLI flags.

    Priority: CLI flags > env vars > config file > defaults
    """
    from kairos.config import get_config
    from kairos.providers.base import ModelConfig  # noqa: F811 — local import for CLI module
    config = get_config()

    # API key
    api_key = (
        os.getenv("KAIROS_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or config.get("providers.deepseek.api_key")
        or config.get("providers.openai.api_key")
    )
    if not api_key:
        return None

    # Base URL
    if not base_url:
        base_url = (
            os.getenv("KAIROS_BASE_URL")
            or config.get("model.base_url")
            or "https://api.deepseek.com"
        )

    # Model name
    if not model_name:
        model_name = (
            os.getenv("KAIROS_MODEL")
            or config.get("model.name")
            or "deepseek-chat"
        )

    return ModelConfig(api_key=api_key, base_url=base_url, model=model_name)


def _get_api_key() -> str | None:
    """Get API key from env or config. (Deprecated — use _get_model_config)"""
    from kairos.config import get_config
    config = get_config()
    return (
        os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("KAIROS_API_KEY")
        or config.get("providers.deepseek.api_key")
        or config.get("providers.openai.api_key")
    )


# ═══════════════════════════════════════════════════════════════
# Permission prompt builder — Rich-based interactive approval
# ═══════════════════════════════════════════════════════════════


def _build_security_for_cli():
    """Build SecurityMiddleware with Rich-based permission prompt for CLI usage.

    Creates a synchronous prompt that uses Rich to ask the user for
    confirmation before executing write/destructive tool calls.
    Supports y/n/(s)ession-level grant.
    """
    import asyncio
    from kairos.middleware.security_mw import SecurityMiddleware
    from kairos.security.permission import PermissionAction, PermissionManager

    pm = PermissionManager(auto_approve=False)

    async def _rich_prompt(request):
        from rich.prompt import Prompt
        from rich.text import Text

        summary = request.summary()
        prompt_text = Text()
        prompt_text.append("🔒 ", style="bold yellow")
        prompt_text.append(summary, style="bold")
        prompt_text.append("\n")
        prompt_text.append("  [y]es  [n]o  [s]ession-grant  [?]", style="dim")

        choice = Prompt.ask(
            prompt_text,
            choices=["y", "n", "s", "?"],
            default="n",
            show_choices=False,
            show_default=False,
        )

        if choice == "?":
            print(f"\n  Tool: {request.tool_name}")
            if request.path:
                print(f"  Path: {request.path}")
            if request.description:
                print(f"  Description: {request.description}")
            # Re-ask
            return await _rich_prompt(request)

        if choice == "y":
            return PermissionAction.ALLOW_ONCE
        if choice == "s":
            return PermissionAction.ALLOW_SESSION
        return PermissionAction.DENY

    pm.set_prompt_callback(_rich_prompt)

    return SecurityMiddleware(permission_manager=pm)


# ═══════════════════════════════════════════════════════════════
# Chat mode — interactive loop with live streaming
# ═══════════════════════════════════════════════════════════════


def _chat_mode(args: list[str], resume: str | None = None, base_url: str | None = None, model_name: str | None = None):
    """Interactive chat loop with Rich TUI and live streaming output."""
    from kairos.core.stateful_agent import StatefulAgent
    from kairos.providers.base import ModelConfig

    model = _get_model_config(base_url=base_url, model_name=model_name)
    if not model:
        console = KairosConsole()
        console.error("No API key found. Set KAIROS_API_KEY or run 'kairos config init'.")
        return

    console = KairosConsole(skin="default", verbose=False, stream=True)
    display_model = getattr(model, "model", "default")
    console.set_status(model=f"{display_model} ({model.base_url})")

    # Interactive chat mode: SecurityMiddleware with Rich-based permission prompt.
    # The user is asked y/n/session for each write/destructive tool call.
    # /yolo toggles auto-approve for all tools.
    _sec_mw = _build_security_for_cli()

    # StatefulAgent for session persistence + streaming support
    agent = StatefulAgent(model=model, middlewares=[_sec_mw])

    # Register delegate_task tool so the agent can spawn sub-agents
    from kairos.agents.delegate import DelegationManager, register_delegate_tool
    from kairos.providers.base import ModelProvider
    dm = DelegationManager(model=ModelProvider(model), config=None)
    register_delegate_tool(dm)

    # Skill manager for skill_manage/skill_view/skills_list tools
    # (already wired via set_skill_manager in loop.py — kept for clarity)

    # Session state
    _last_input: list[str] = [""]  # for /retry (mutable for closure capture)
    _goal: dict = {"text": "", "active": False}  # for /goal
    _show_reasoning: list[bool] = [True]  # for /reasoning (mutable)

    # Resume session if requested
    if resume:
        if agent.load_session(resume):
            console.success(f"Resumed session: {resume}")
            console.console.print()
        else:
            console.error("Session not found. Check the name with --list-sessions.", 
                          hint="Use: kairos --list-sessions")
            return

    from kairos import __version__
    from kairos.tools.registry import get_all_tools
    session_count = len(agent.list_sessions())
    tool_count = len(get_all_tools())

    # ── Welcome panel ────────────────────────────────────────
    console.show_welcome(
        version=__version__,
        model=display_model,
        base_url=model.base_url,
        session_count=session_count,
        tool_count=tool_count,
    )

    while True:
        try:
            user_input = console.prompt("You")
        except (KeyboardInterrupt, EOFError):
            console.console.print("\n👋 Goodbye!")
            break

        if not user_input.strip():
            continue

        if user_input.startswith("/"):
            _handle_slash(console, user_input, agent, model, _last_input_ref, _goal, _show_reasoning)
            continue

        _last_input_ref[0] = user_input  # save for /retry
        console.user_input(user_input)

        # ── Inject active goal into system prompt ────────────
        if _goal["active"] and _goal["text"] and hasattr(agent, "system_prompt"):
            goal_block = f"\n<goal>\nCurrent goal: {_goal['text']}\nWork towards this goal across turns.\n</goal>"
            if goal_block not in agent.system_prompt:
                agent.system_prompt += goal_block

        try:
            # ── Streaming mode: use chat_stream() ────────────
            stream = agent.chat_stream(user_input)
            final_content, final_event = console.stream_response(stream)

            # Show tool calls (always a one-liner, tree in verbose)
            if final_event and final_event.get("tool_calls"):
                for tc in final_event["tool_calls"]:
                    console.tool_call(
                        name=tc.get("name", "?"),
                        args=tc.get("args", tc.get("arguments", {})),
                        result=tc.get("result", ""),
                        duration_ms=tc.get("duration_ms", 0),
                    )

            # Show token usage
            if final_event and final_event.get("usage"):
                console.show_usage(final_event["usage"])

            # Skills are created by the Agent itself via skill_manage tool
            # — aligned with Hermes: the LLM decides when to save, not a heuristic engine.

        except KeyboardInterrupt:
            console.console.print("\n[yellow]⏸️  Interrupted[/]")
            if hasattr(agent, "interrupt"):
                agent.interrupt()
            continue
        except Exception as e:
            err_msg = str(e)
            if "timeout" in err_msg.lower():
                hint = "The operation took too long. Try a simpler query or increase timeout."
            elif "connection" in err_msg.lower():
                hint = "Cannot reach the model server. Check your network and --base-url."
            elif "api key" in err_msg.lower() or "auth" in err_msg.lower():
                hint = "API key issue. Check KAIROS_API_KEY or run 'kairos config init'."
            elif "rate limit" in err_msg.lower() or "429" in err_msg:
                hint = "Rate limited. Wait a moment and try again."
            else:
                hint = "Try /help to see available commands."
            console.error(f"Agent error: {err_msg}", hint=hint)
            continue

        console.console.print()


# ═══════════════════════════════════════════════════════════════
# Run mode — one-shot query
# ═══════════════════════════════════════════════════════════════


def _run_mode(args: list[str], base_url: str | None = None, model_name: str | None = None):
    """Single query mode with streaming."""
    from kairos.core.stateful_agent import StatefulAgent
    from kairos.providers.base import ModelConfig

    if not args:
        console = KairosConsole()
        console.error("Usage: kairos run <query>")
        return

    model = _get_model_config(base_url=base_url, model_name=model_name)
    if not model:
        console = KairosConsole()
        console.error("No API key found. Set KAIROS_API_KEY or DEEPSEEK_API_KEY.")
        return

    query = " ".join(args)
    agent = StatefulAgent(model=model)
    console = KairosConsole(stream=True)

    try:
        stream = agent.chat_stream(query)
        final_content, final_event = console.stream_response(stream)

        if final_event and final_event.get("usage"):
            console.show_usage(final_event["usage"])
    except KeyboardInterrupt:
        console.console.print("\n[yellow]⏸️  Interrupted[/]")
    except Exception as e:
        console.error(f"Error: {e}")


# ═══════════════════════════════════════════════════════════════
# Cron mode
# ═══════════════════════════════════════════════════════════════


def _cron_mode(args: list[str]):
    """Cron scheduler management."""
    from kairos.cron.scheduler import CronScheduler, Job, CronSchedule

    console = KairosConsole()
    scheduler = CronScheduler()

    if not args or args[0] == "list":
        jobs = scheduler.list()
        console.show_cron_jobs(jobs)
    elif args[0] == "add":
        if len(args) < 3:
            console.error("Usage: kairos cron add <name> <cron_expr>")
            return
        name = args[1]
        expr_parts = args[2].split()
        if len(expr_parts) != 5:
            console.error("Cron expression must have 5 fields: min hour dom month dow")
            return
        try:
            schedule = CronSchedule(
                minute=_parse_field(expr_parts[0]),
                hour=_parse_field(expr_parts[1]),
                day=_parse_field(expr_parts[2]),
                month=_parse_field(expr_parts[3]),
                weekday=_parse_field(expr_parts[4]),
            )
        except ValueError as e:
            console.error(f"Invalid cron field: {e}")
            return
        job = Job(name=name, schedule=schedule)
        scheduler.register(job)
        console.success(f"Registered cron job: {job.id} ({name})")
    elif args[0] == "pause" and len(args) >= 2:
        if job := scheduler.pause(args[1]):
            console.success(f"Paused job: {job.name}")
        else:
            console.error(f"Job not found: {args[1]}")
    elif args[0] == "resume" and len(args) >= 2:
        if job := scheduler.resume(args[1]):
            console.success(f"Resumed job: {job.name}")
        else:
            console.error(f"Job not found: {args[1]}")
    elif args[0] == "cancel" and len(args) >= 2:
        if job := scheduler.cancel(args[1]):
            console.success(f"Cancelled job: {job.name}")
        else:
            console.error(f"Job not found: {args[1]}")
    elif args[0] == "remove" and len(args) >= 2:
        scheduler.remove(args[1])
        console.success(f"Removed job: {args[1]}")
    elif args[0] == "run" and len(args) >= 2:
        if job := scheduler.run_now(args[1]):
            console.success(f"Triggered job: {job.name} (will fire on next tick)")
        else:
            console.error(f"Job not found: {args[1]}")
    else:
        console.error(f"Unknown cron command: {' '.join(args)}")
        console.info("Usage: kairos cron [list|add|pause|resume|run|cancel|remove]")


# ═══════════════════════════════════════════════════════════════
# Config / Skill / Curator modes
# ═══════════════════════════════════════════════════════════════


def _config_mode(args: list[str]):
    """Config management."""
    from kairos.config import write_default_config, get_config

    console = KairosConsole()
    if not args or args[0] == "init":
        path = args[1] if len(args) > 1 else "~/.config/kairos/config.yaml"
        write_default_config(path)
    elif args[0] == "show":
        cfg = get_config()
        if cfg.path:
            console.info(f"Config: {cfg.path}")
        else:
            console.info("No config file found. Run 'kairos config init' to create one.")
    elif args[0] == "migrate":
        _migrate_config()
    else:
        console.error(f"Unknown config command: {args[0]}")
        console.info("Usage: kairos config [init|show|migrate]")


def _migrate_config():
    """Upgrade old config files with new defaults."""
    import yaml
    from pathlib import Path

    config_path = Path(os.path.expanduser("~/.config/kairos/config.yaml"))
    if not config_path.exists():
        print("No config file found. Run 'kairos config init' first.")
        return

    current = yaml.safe_load(config_path.read_text()) or {}

    # Default values that may be missing in older configs
    defaults = {
        "agent": {"max_turns": 90, "tool_use_enforcement": True},
        "compression": {"enabled": True, "threshold": 0.5, "target_ratio": 0.2},
        "display": {"skin": "default", "tool_progress": True},
        "delegation": {"max_concurrent_children": 3, "max_iterations": 50},
        "security": {"sandbox_audit": True, "block_high_risk": True},
        "memory": {"max_injection_tokens": 2000},
    }

    added = 0
    for section, values in defaults.items():
        if section not in current:
            current[section] = values
            added += len(values)
            print(f"  + Added [{section}] section")
        elif isinstance(values, dict) and isinstance(current[section], dict):
            for key, val in values.items():
                if key not in current[section]:
                    current[section][key] = val
                    added += 1
                    print(f"  + Added {section}.{key} = {val}")

    if added:
        # Backup old config
        backup_path = config_path.with_suffix(".yaml.bak")
        import shutil
        shutil.copy2(config_path, backup_path)
        config_path.write_text(yaml.dump(current, default_flow_style=False, allow_unicode=True))
        print(f"\n✅ Migrated {added} new keys. Backup saved to {backup_path}")
    else:
        print("✅ Config is up to date. Nothing to migrate.")


def _skill_mode(args: list[str]):
    """Skill management."""
    from kairos.skills.manager import SkillManager
    from kairos.skills.marketplace import SkillMarketplace

    manager = SkillManager()
    marketplace = SkillMarketplace(manager)

    if not args or args[0] == "list":
        manager.scan()
        skills = manager.list_skills()
        if not skills:
            print("No skills installed. Use 'kairos skill install <source>' to add one.")
            return
        stats = manager.stats()
        print(f"Installed skills ({stats['active']} active, {stats['stale']} stale):\n")
        for entry in skills:
            icon = "●" if entry.status.value == "active" else "○" if entry.status.value == "stale" else "✕"
            print(f"  {icon} {entry.name}")
            if entry.description:
                print(f"      {entry.description[:80]}")
        print()
    elif args[0] == "view" and len(args) >= 2:
        content = manager.get_skill_content(args[1])
        if content is None:
            print(f"Error: Skill '{args[1]}' not found.")
            return
        print(f"Skill: {content['name']}")
        print(f"Status: {content['status']} | Uses: {content['use_count']}")
        if content.get('description'):
            print(f"Description: {content['description']}")
        print(f"\n--- BEGIN SKILL ---")
        print(content['content'])
        print(f"--- END SKILL ---")
    elif args[0] == "install" and len(args) >= 2:
        source = args[1]
        name = args[2] if len(args) > 2 else None
        print(f"Installing from: {source} ...")
        result = marketplace.install(source, name)
        if result.get("success"):
            print(f"✅ Installed: {result['name']} → {result['path']}")
        else:
            print(f"❌ Error: {result.get('error')}")
    elif args[0] == "uninstall" and len(args) >= 2:
        result = marketplace.uninstall(args[1])
        if result.get("success"):
            print(f"✅ Uninstalled: {result['name']}")
        else:
            print(f"❌ Error: {result.get('error')}")
    elif args[0] == "update":
        if len(args) >= 2:
            result = marketplace.update(args[1])
        else:
            results = marketplace.update_all()
            for r in results:
                status = "✅" if r.get("success") else "❌"
                print(f"  {status} {r.get('name', '?')}: {r.get('error', 'updated')}")
            return
        if result.get("success"):
            print(f"✅ Updated: {result['name']}")
        else:
            print(f"❌ Error: {result.get('error')}")
    elif args[0] == "marketplace":
        skills = marketplace.list_marketplace()
        if not skills:
            print("No marketplace skills installed.")
            return
        print(f"Marketplace skills ({len(skills)}):\n")
        for s in skills:
            print(f"  ● {s['name']} ({s.get('version', '?')})")
            print(f"    source: {s.get('source', '?')}")
            if s.get('description'):
                print(f"    {s['description'][:80]}")
    else:
        print(f"Unknown skill command: {' '.join(args)}")
        print("Usage: kairos skill [list|view|install|uninstall|update|marketplace]")


def _curator_mode(args: list[str]):
    """Curator lifecycle management."""
    from kairos.skills.manager import SkillManager

    manager = SkillManager()
    if not args or args[0] == "status":
        stats = manager.stats()
        print("Curator Status\n" + "=" * 40)
        print(f"  Total skills:   {stats['total']}")
        print(f"  Active:         {stats['active']}")
        print(f"  Stale (>30d):   {stats['stale']}")
        print(f"  Archived:       {stats['archived']}")
        print(f"  Categories:     {stats['categories']}")
        print(f"  Backups:        {stats['backups']}")
        stale = [e for e in manager.list_skills() if e.status.value == "stale"]
        if stale:
            print(f"\nStale skills (unused >30 days):")
            for s in stale:
                last = "never" if s.last_used_at is None else f"{int(time.time() - s.last_used_at) // 86400}d ago"
                print(f"  ○ {s.name} (last used: {last})")
    elif args[0] == "clean":
        days = int(args[1]) if len(args) > 1 else None
        result = manager.clean(days)
        print(f"Cleaned {result['cleaned']} archived entries ({result['freed_bytes']} bytes freed)")
    elif args[0] == "reindex":
        result = manager.reindex()
        print(f"Reindexed: +{result['added']} added, -{result['removed']} removed")
    else:
        print(f"Unknown curator command: {' '.join(args)}")
        print("Usage: kairos curator [status|clean|reindex]")


# ═══════════════════════════════════════════════════════════════
# Doctor — health check
# ═══════════════════════════════════════════════════════════════


def _doctor_mode():
    """Run a health check on the Kairos installation."""
    import importlib
    from pathlib import Path
    from kairos import __version__

    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = ""):
        results.append((name, ok, detail))

    # 1. Python version
    py_ver = sys.version_info
    check("Python 3.10+", py_ver >= (3, 10), f"{py_ver.major}.{py_ver.minor}.{py_ver.micro}")

    # 2. Config file
    config_path = Path(os.path.expanduser("~/.config/kairos/config.yaml"))
    config_ok = config_path.exists()
    check("Config file", config_ok, str(config_path) if config_ok else "Not found")

    # 3. API key
    api_key = os.getenv("KAIROS_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    check("API key", bool(api_key), "Set" if api_key else "Missing — set KAIROS_API_KEY or DEEPSEEK_API_KEY")

    # 4. Skills directory
    skills_dir = Path(os.path.expanduser("~/.kairos/skills"))
    check("Skills directory", skills_dir.exists(), str(skills_dir))

    # 5. Core dependencies
    for pkg in ("openai", "rich", "pydantic", "aiohttp"):
        try:
            importlib.import_module(pkg)
            check(f"Dependency: {pkg}", True, "Installed")
        except ImportError:
            check(f"Dependency: {pkg}", False, "Not installed — pip install")

    # 6. Model reachability (quick test — only if API key set)
    if api_key:
        try:
            from kairos.providers.base import ModelConfig, ModelProvider
            base = os.getenv("KAIROS_BASE_URL", "https://api.deepseek.com/v1")
            model = os.getenv("KAIROS_MODEL", "deepseek-chat")
            provider = ModelProvider(ModelConfig(api_key=api_key, base_url=base, model=model))
            test_result = provider.chat([{"role": "user", "content": "ping"}], max_tokens=5)
            ok = "error" not in str(test_result).lower()[:100]
            check("Model reachable", ok, f"{model} @ {base}" if ok else "Connection failed")
        except Exception as e:
            check("Model reachable", False, str(e)[:80])

    # Print report
    print(f"\n🩺 Kairos Doctor — v{__version__}\n{'=' * 50}")
    all_ok = True
    for name, ok, detail in results:
        icon = "✅" if ok else "❌"
        if not ok:
            all_ok = False
        detail_str = f" — {detail}" if detail else ""
        print(f"  {icon} {name}{detail_str}")

    print()
    if all_ok:
        print("✨ All checks passed. Kairos is healthy!")
    else:
        print("⚠️  Some checks failed. Review the ❌ items above.")
    print()


# ═══════════════════════════════════════════════════════════════
# Web UI mode
# ═══════════════════════════════════════════════════════════════


def _web_mode(base_url: str | None = None, model_name: str | None = None):
    """Start the Kairos Web UI server."""
    import asyncio

    from kairos.web import WebServer
    from kairos.core.stateful_agent import StatefulAgent

    model = _get_model_config(base_url=base_url, model_name=model_name)
    if not model:
        print("❌ No API key found. Set KAIROS_API_KEY or DEEPSEEK_API_KEY.")
        return

    agent = StatefulAgent(model=model)
    server = WebServer(agent=agent, host="127.0.0.1", port=8080)

    async def run():
        await server.start()
        print(f"\n✨ Kairos Web UI: http://127.0.0.1:8080\n")
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")


# ═══════════════════════════════════════════════════════════════
# TUI mode — Textual-powered terminal UI
# ═══════════════════════════════════════════════════════════════


def _tui_mode(argv: list[str] | None = None) -> None:
    """Launch the Textual TUI.

    Delegates to kairos.tui.__main__.main() for arg parsing and app creation.
    """
    from kairos.tui.__main__ import main as tui_main

    # Pass only the args after 'tui'
    if argv and "tui" in argv:
        idx = argv.index("tui")
        tui_args = argv[idx + 1:]
    else:
        tui_args = []

    tui_main(tui_args)


# ═══════════════════════════════════════════════════════════════
# Slash command handler
# ═══════════════════════════════════════════════════════════════


def _parse_field(field: str) -> list[int]:
    if field == "*":
        return []
    if field.startswith("*/"):
        step = int(field[2:])
        return list(range(0, 60, step))
    parts = field.split(",")
    values = []
    for p in parts:
        if "-" in p:
            lo, hi = p.split("-")
            values.extend(range(int(lo), int(hi) + 1))
        else:
            values.append(int(p))
    return values


def _handle_slash(console, cmd: str, agent, model_config, last_input_ref=None, goal=None, show_reasoning_ref=None):
    """Handle slash commands in chat mode."""
    parts = cmd.strip().split()
    command = parts[0].lower()

    if command in ("/exit", "/quit"):
        console.console.print("👋 Goodbye!")
        sys.exit(0)

    elif command == "/help":
        console.show_help()

    elif command == "/history":
        console.show_history()

    elif command == "/clear":
        console._history.clear()
        console.success("Conversation history cleared.")

    elif command == "/retry":
        last = (last_input_ref or [""])[0]
        if not last:
            console.error("Nothing to retry. Send a message first.")
            return
        console.info(f"Retrying: {last[:80]}{'...' if len(last) > 80 else ''}")
        console.user_input(last)
        try:
            stream = agent.chat_stream(last)
            final_content, final_event = console.stream_response(stream)
            if final_event and final_event.get("usage"):
                console.show_usage(final_event["usage"])
        except Exception as e:
            console.error(f"Error: {e}")

    elif command == "/undo":
        # Remove last exchange from console history
        removed = 0
        while console._history:
            entry = console._history[-1]
            if entry.get("role") in ("user", "agent"):
                console._history.pop()
                removed += 1
                if entry.get("role") == "user":
                    break
            else:
                console._history.pop()
                removed += 1
        if removed:
            console.success(f"Undid last exchange ({removed} messages removed).")
            # Tell agent to pop last exchange
            if hasattr(agent, "pop_last_exchange"):
                agent.pop_last_exchange()
            if last_input_ref:
                last_input_ref[0] = ""  # clear retry target
        else:
            console.info("Nothing to undo.")

    elif command == "/verbose":
        console.verbose = not console.verbose
        console.info(f"Verbose tool output: {'ON' if console.verbose else 'OFF'}")

    elif command == "/skin":
        if len(parts) >= 2:
            console.set_skin(parts[1])
        else:
            console.info(
                f"Current skin: {console.skin_name}. "
                "Available: default, hacker, retro, minimal"
            )

    elif command == "/tools":
        console.show_tools()

    elif command == "/cron":
        from kairos.cron.scheduler import CronScheduler
        scheduler = CronScheduler()
        console.show_cron_jobs(scheduler.list())

    elif command == "/model":
        if len(parts) >= 2:
            model_config.model = parts[1]
            console.set_status(model=parts[1])
            console.success(f"Model set to: {parts[1]}")
        else:
            console.info(f"Current model: {getattr(model_config, 'model', 'default')}")

    elif command == "/save":
        if len(parts) >= 2:
            if hasattr(agent, "save_session"):
                agent.save_session(parts[1])
                console.success(f"Session saved as: {parts[1]}")
                console.set_status(session=parts[1])
            else:
                console.error("Current agent doesn't support session persistence.")
        else:
            console.error("Usage: /save <name>")

    elif command == "/sessions":
        if hasattr(agent, "list_sessions"):
            sessions = agent.list_sessions()
            if not sessions:
                console.info("No saved sessions.")
            else:
                console.console.print(f"\n[bold]Saved sessions ({len(sessions)}):[/]")
                for s in sessions[:10]:
                    ts = time.strftime("%m-%d %H:%M", time.localtime(s.get("saved_at", 0)))
                    console.console.print(
                        f"  ● [cyan]{s['name']}[/] "
                        f"({s.get('message_count',0)} msgs, "
                        f"{s.get('turn_count',0)} turns, {ts})"
                    )
        else:
            console.error("Session listing not available.")

    elif command == "/session":
        if len(parts) < 3:
            console.error("Usage: /session rename <old> <new> | /session delete <name>")
            return
        sub = parts[1].lower()
        if sub == "rename" and len(parts) >= 4:
            old, new = parts[2], parts[3]
            if hasattr(agent, "rename_session"):
                if agent.rename_session(old, new):
                    console.success(f"Session renamed: {old} → {new}")
                    if new == getattr(agent, "_session_id", None):
                        console.set_status(session=new)
                else:
                    console.error(f"Session not found: {old}")
            else:
                console.error("Session renaming not available.")
        elif sub == "delete":
            name = parts[2]
            if hasattr(agent, "delete_session"):
                if agent.delete_session(name):
                    console.success(f"Session deleted: {name}")
                else:
                    console.error(f"Session not found: {name}")
            else:
                console.error("Session deletion not available.")
        else:
            console.error(f"Unknown session command: {sub}")

    elif command == "/run":
        if len(parts) >= 2:
            query = " ".join(parts[1:])
            console.user_input(query)
            try:
                stream = agent.chat_stream(query)
                final_content, final_event = console.stream_response(stream)
                if final_event and final_event.get("usage"):
                    console.show_usage(final_event["usage"])
            except Exception as e:
                console.error(f"Error: {e}")
        else:
            console.error("Usage: /run <query>")

    elif command == "/skills":
        from kairos.skills.manager import SkillManager
        manager = SkillManager()
        manager.scan()
        skills = manager.list_skills()
        if not skills:
            console.info("No skills available. Use 'kairos skill install <source>'.")
        else:
            for entry in skills:
                icon = "●" if entry.status.value == "active" else "○" if entry.status.value == "stale" else "✕"
                line = f"  {icon} {entry.name}"
                if entry.description:
                    line += f" — {entry.description[:60]}"
                console.info(line)

    elif command == "/perm":
        _handle_perm_slash(console, parts, agent)

    elif command == "/yolo":
        new_state = console.toggle_yolo()
        if new_state:
            console.success(
                "⚡ YOLO mode ON — all safety checks bypassed. "
                "Use /yolo again to disable."
            )
            # Bypass SandboxAudit and PermissionManager
            _set_yolo_on_middleware(agent, True)
        else:
            console.info("🛡️  YOLO mode OFF — safety checks and permission prompts restored.")
            _set_yolo_on_middleware(agent, False)

    elif command == "/goal":
        _handle_goal(console, parts, goal)

    elif command == "/reasoning":
        if show_reasoning_ref is not None:
            if len(parts) >= 2 and parts[1] in ("on", "off"):
                show_reasoning_ref[0] = parts[1] == "on"
            else:
                show_reasoning_ref[0] = not show_reasoning_ref[0]
            console.info(f"Reasoning display: {'ON 🧠' if show_reasoning_ref[0] else 'OFF'}")

    elif command == "/background":
        if len(parts) < 2:
            console.error("Usage: /background <prompt>")
            return
        prompt = " ".join(parts[1:])
        _run_background(console, agent, prompt)

    elif command == "/edit":
        content = console.multiline_prompt("edit")
        if not content.strip():
            console.info("Cancelled.")
            return
        console.user_input(content)
        try:
            stream = agent.chat_stream(content)
            final_content, final_event = console.stream_response(stream)
            if final_event and final_event.get("usage"):
                console.show_usage(final_event["usage"])
        except Exception as e:
            console.error(f"Error: {e}")

    else:
        console.error(f"Unknown command: {command}. Type /help for available commands.")


def _set_yolo_on_middleware(agent, enabled: bool) -> None:
    """Set YOLO bypass on SandboxAudit middleware and PermissionManager if present."""
    for mw in getattr(agent, "_middlewares", []):
        if hasattr(mw, "yolo_bypass"):
            mw.yolo_bypass = enabled
        if hasattr(mw, "permission_manager"):
            mw.permission_manager.set_auto_approve(enabled)


def _handle_goal(console, parts: list[str], goal: dict) -> None:
    """Handle /goal command: set, status, pause, resume, clear."""
    if len(parts) < 2:
        if goal["active"]:
            console.info(f"🎯 Active goal: {goal['text']}")
            console.info("  /goal status | /goal pause | /goal clear")
        else:
            console.info("No active goal. Use /goal <text> to set one.")
        return

    sub = parts[1].lower()
    if sub == "status":
        if goal["active"]:
            console.info(f"🎯 Goal (active): {goal['text']}")
        else:
            console.info("No active goal.")
    elif sub == "pause":
        goal["active"] = False
        console.success(f"⏸️  Goal paused: {goal['text']}")
    elif sub == "resume":
        if goal["text"]:
            goal["active"] = True
            console.success(f"▶️  Goal resumed: {goal['text']}")
        else:
            console.error("No goal to resume. Set one with /goal <text>")
    elif sub == "clear":
        old = goal["text"]
        goal["text"] = ""
        goal["active"] = False
        console.success(f"🗑️  Goal cleared: {old}")
    else:
        # Set new goal
        text = " ".join(parts[1:])
        goal["text"] = text
        goal["active"] = True
        console.success(f"🎯 Goal set: {text}")
        console.info("The agent will work towards this goal across turns. Use /goal status to check.")


_BG_TASKS: dict[int, dict] = {}  # task_id -> {prompt, thread, done}


def _run_background(console, agent, prompt: str) -> None:
    """Run a prompt in a background thread. Notifies when done."""
    import threading

    task_id = len(_BG_TASKS) + 1
    console.info(f"📋 Background task #{task_id} started: {prompt[:60]}...")

    def worker():
        try:
            result = agent.chat(prompt)
            content = result.get("content", str(result))
            _BG_TASKS[task_id]["done"] = True
            _BG_TASKS[task_id]["result"] = content
            # Print notification
            preview = content[:200].replace("\n", " ")
            console.console.print(
                f"\n[bold green]✅ BG #{task_id} done:[/] {preview}..."
                if len(content) > 200 else
                f"\n[bold green]✅ BG #{task_id} done:[/] {preview}"
            )
        except Exception as e:
            _BG_TASKS[task_id]["done"] = True
            _BG_TASKS[task_id]["error"] = str(e)
            console.console.print(f"\n[bold red]❌ BG #{task_id} failed:[/] {e}")

    thread = threading.Thread(target=worker, daemon=True)
    _BG_TASKS[task_id] = {"prompt": prompt, "thread": thread, "done": False}
    thread.start()


def _handle_perm_slash(console, parts: list[str], agent):
    """Handle /perm commands."""
    from kairos.security.permission import PermissionLevel, ToolPolicy

    sec_mw = None
    for mw in getattr(agent, "_middlewares", []):
        if hasattr(mw, "permission_manager"):
            sec_mw = mw
            break

    if not sec_mw:
        console.error("Security middleware not active.")
        return

    perm = sec_mw.permission_manager

    if len(parts) < 2 or parts[1] == "show":
        console.console.print("\n[bold]Permission Policies:[/]")
        for pattern, policy in sorted(perm._policies.items()):
            icon = {"block": "⛔", "ask": "⚠️", "trust": "✅"}.get(policy.level.value, "❓")
            console.console.print(f"  {icon} [cyan]{pattern}[/] → {policy.level.value}")
        console.console.print(f"\nDefault: [dim]{perm._default_policy.level.value}[/]")
        return

    action = parts[1]
    if action in ("trust", "ask", "block") and len(parts) >= 3:
        tool_pattern = parts[2]
        level = PermissionLevel(action)
        perm.set_policy(ToolPolicy(tool_pattern, level=level))
        console.success(f"Set {tool_pattern} → {action}")
    else:
        console.error("Usage: /perm show | /perm trust <tool> | /perm ask <tool> | /perm block <tool>")


if __name__ == "__main__":
    main()
