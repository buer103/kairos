"""CLI entry point for Kairos — Rich TUI, chat, cron, and config commands.

Usage (Hermes-style):
    kairos                    Interactive chat mode (default) with live streaming
    kairos <query>            One-shot query (no 'run' needed)
    kairos chat               Explicit chat mode
    kairos run <query>        One-shot query (explicit)
    kairos --resume <name>    Resume a saved session
    kairos --list-sessions    List saved sessions
    kairos --version          Show version
    kairos cron|config|skill|curator  Management subcommands
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
# Chat mode — interactive loop with live streaming
# ═══════════════════════════════════════════════════════════════


def _chat_mode(args: list[str], resume: str | None = None, base_url: str | None = None, model_name: str | None = None):
    """Interactive chat loop with Rich TUI and live streaming output."""
    from kairos.core.stateful_agent import StatefulAgent
    from kairos.providers.base import ModelConfig
    from kairos.security.permission import PermissionManager, PermissionAction, PermissionRequest
    from kairos.middleware.security_mw import SecurityMiddleware

    model = _get_model_config(base_url=base_url, model_name=model_name)
    if not model:
        console = KairosConsole()
        console.error("No API key found. Set KAIROS_API_KEY or run 'kairos config init'.")
        return

    console = KairosConsole(skin="default", verbose=False, stream=True)
    display_model = getattr(model, "model", "default")
    console.set_status(model=f"{display_model} ({model.base_url})")

    # Permission manager with interactive prompt
    perm = PermissionManager()
    _perm_console = console

    async def _prompt_user(request: PermissionRequest) -> PermissionAction | None:
        _perm_console.console.print()
        _perm_console.console.print(
            f"🔐 [bold yellow]Permission Required[/] — [cyan]{request.tool_name}[/]"
        )
        if request.description:
            _perm_console.console.print(f"   {request.description}")
        if request.path:
            _perm_console.console.print(f"   Path: [dim]{request.path}[/]")
        _perm_console.console.print()
        _perm_console.console.print(
            "  [green][a][/] Allow once   [yellow][y][/] Always allow   [red][n][/] Deny"
        )
        try:
            choice = _perm_console.prompt("Permission")
            key = choice.strip().lower()[:1] if choice else "n"
            return {"a": PermissionAction.ALLOW_ONCE, "y": PermissionAction.ALLOW_SESSION}.get(
                key, PermissionAction.DENY)
        except (KeyboardInterrupt, EOFError):
            return PermissionAction.DENY

    perm.set_prompt_callback(_prompt_user)
    sec_mw = SecurityMiddleware(permission_manager=perm)

    # StatefulAgent for session persistence + streaming support
    agent = StatefulAgent(model=model, middlewares=[sec_mw])

    # Resume session if requested
    if resume:
        if agent.load_session(resume):
            console.success(f"Resumed session: {resume}")
            console.console.print()
        else:
            console.error(f"Session not found: {resume}")
            return

    from kairos import __version__
    session_count = len(agent.list_sessions())
    console.info(
        f"Kairos {__version__} — {session_count} saved session(s) — type /help for commands"
    )
    console.info(f"Model: {display_model} @ {model.base_url}  ·  Streaming: ON")
    console.console.print()

    while True:
        try:
            user_input = console.prompt("Kairos")
        except (KeyboardInterrupt, EOFError):
            console.console.print("\n👋 Goodbye!")
            break

        if not user_input.strip():
            continue

        if user_input.startswith("/"):
            _handle_slash(console, user_input, agent, model)
            continue

        console.user_input(user_input)

        try:
            # ── Streaming mode: use chat_stream() ────────────
            stream = agent.chat_stream(user_input)
            final_content, final_event = console.stream_response(stream)

            # Show verbose tool calls if enabled
            if console.verbose and final_event and final_event.get("tool_calls"):
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

        except KeyboardInterrupt:
            console.console.print("\n[yellow]⏸️  Interrupted[/]")
            if hasattr(agent, "interrupt"):
                agent.interrupt()
            continue
        except Exception as e:
            console.error(f"Agent error: {e}")
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
    else:
        console.error(f"Unknown cron command: {' '.join(args)}")
        console.info("Usage: kairos cron [list|add|pause|resume|cancel|remove]")


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
    else:
        console.error(f"Unknown config command: {args[0]}")
        console.info("Usage: kairos config [init|show]")


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


def _handle_slash(console, cmd: str, agent, model_config):
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

    else:
        console.error(f"Unknown command: {command}. Type /help for available commands.")


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
