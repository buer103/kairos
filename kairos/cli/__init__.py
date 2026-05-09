"""CLI entry point for Kairos — Rich TUI, chat, cron, and config commands."""

from __future__ import annotations

import os
import sys
import time

from kairos.cli.rich_ui import KairosConsole, SKINS

__all__ = ["KairosConsole", "SKINS", "main"]


def main():
    """Main entry point: kairos chat / kairos run / kairos cron / kairos config / kairos skill / kairos curator."""
    args = sys.argv[1:]

    if not args:
        _print_usage()
        return

    if args[0] == "--version":
        from kairos import __version__
        print(f"kairos {__version__}")
        return

    if args[0] == "chat":
        _chat_mode(args[1:])
    elif args[0] == "run":
        _run_mode(args[1:])
    elif args[0] == "cron":
        _cron_mode(args[1:])
    elif args[0] == "config":
        _config_mode(args[1:])
    elif args[0] == "skill":
        _skill_mode(args[1:])
    elif args[0] == "curator":
        _curator_mode(args[1:])
    else:
        print(f"Unknown command: {args[0]}")
        _print_usage()


def _print_usage():
    print("Kairos — The right tool, at the right moment.")
    print()
    print("Usage:")
    print("  kairos chat                Interactive chat mode (Rich TUI)")
    print("  kairos run <query>         Single query mode")
    print("  kairos cron                Cron scheduler management")
    print("  kairos config init         Generate default config file")
    print("  kairos skill list          List installed skills")
    print("  kairos skill install <url> Install a skill from GitHub/URL/local")
    print("  kairos skill view <name>   View a skill's content")
    print("  kairos curator status      Show skill lifecycle status")
    print("  kairos curator clean       Remove archived skills older than 90 days")
    print("  kairos --version           Show version")


def _chat_mode(args: list[str]):
    """Interactive chat loop with Rich TUI."""
    from kairos import Agent
    from kairos.providers.base import ModelConfig
    from kairos.config import get_config

    config = get_config()
    api_key = (
        os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or config.get("providers.deepseek.api_key")
        or config.get("providers.openai.api_key")
    )
    if not api_key:
        console = KairosConsole()
        console.error("No API key found. Set DEEPSEEK_API_KEY or run 'kairos config init'.")
        return

    console = KairosConsole(skin="default", verbose=False)
    model = ModelConfig(api_key=api_key)
    agent = Agent(model=model)

    console.info("Kairos chat — type /help for available commands")
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
        console.spinner_start("Thinking...")

        try:
            result = agent.run(user_input)
        except Exception as e:
            console.spinner_stop()
            console.error(f"Agent error: {e}")
            continue

        console.spinner_update("Formatting response...")
        console.spinner_stop()

        console.agent_output(
            content=result.get("content", "(no response)"),
            confidence=result.get("confidence"),
        )

        if console.verbose and result.get("evidence"):
            for step in result["evidence"]:
                console.tool_call(
                    name=step.get("tool", "?"),
                    args=step.get("args", {}),
                    result=step.get("result", ""),
                    duration_ms=step.get("duration_ms", 0),
                )

        console.console.print()


def _run_mode(args: list[str]):
    """Single query mode."""
    from kairos import Agent
    from kairos.providers.base import ModelConfig
    from kairos.config import get_config

    if not args:
        console = KairosConsole()
        console.error("Usage: kairos run <query>")
        return

    config = get_config()
    api_key = (
        os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or config.get("providers.deepseek.api_key")
        or config.get("providers.openai.api_key")
    )
    if not api_key:
        console = KairosConsole()
        console.error("No API key found.")
        return

    query = " ".join(args)
    model = ModelConfig(api_key=api_key)
    agent = Agent(model=model)

    console = KairosConsole(verbose=False)
    console.spinner_start("Processing...")
    try:
        result = agent.run(query)
        console.spinner_stop()
        console.agent_output(
            content=result.get("content", ""),
            confidence=result.get("confidence"),
        )
    except Exception as e:
        console.spinner_stop()
        console.error(f"Error: {e}")


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


def _config_mode(args: list[str]):
    """Config management: kairos config init / show."""
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
    """Skill management: kairos skill list|view|install|uninstall|update."""
    from kairos.skills.manager import SkillManager
    from kairos.skills.marketplace import SkillMarketplace

    manager = SkillManager()
    marketplace = SkillMarketplace(manager)

    if not args or args[0] == "list":
        # Scan first to pick up any new skills
        manager.scan()
        skills = manager.list_skills()
        if not skills:
            print("No skills installed. Use 'kairos skill install <source>' to add one.")
            return

        print(f"Installed skills ({manager.stats()['active']} active, {manager.stats()['stale']} stale):\n")
        for entry in skills:
            status_icon = "●" if entry.status.value == "active" else "○" if entry.status.value == "stale" else "✕"
            print(f"  {status_icon} {entry.name}")
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
        linked = {k: v for k, v in content.get('linked_files', {}).items() if v}
        if linked:
            print(f"\nLinked files:")
            for kind, files in linked.items():
                for f in files:
                    print(f"  {kind}/{f}")
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
    """Curator lifecycle management: kairos curator status|clean|reindex."""
    from kairos.skills.manager import SkillManager
    import json

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

        # Show stale skills
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
            console.success(f"Model set to: {parts[1]}")
        else:
            console.info(f"Current model: {getattr(model_config, 'model', 'default')}")

    elif command == "/run":
        if len(parts) >= 2:
            query = " ".join(parts[1:])
            console.user_input(query)
            console.spinner_start("Thinking...")
            try:
                result = agent.run(query)
                console.spinner_stop()
                console.agent_output(
                    content=result.get("content", ""),
                    confidence=result.get("confidence"),
                )
            except Exception as e:
                console.spinner_stop()
                console.error(f"Error: {e}")
        else:
            console.error("Usage: /run <query>")

    elif command == "/skills":
        from kairos.skills.manager import SkillManager
        manager = SkillManager()
        manager.scan()
        skills = manager.list_skills()
        if not skills:
            console.info("No skills available. Use 'kairos skill install <source>' to add one.")
        else:
            for entry in skills:
                status_icon = "●" if entry.status.value == "active" else "○" if entry.status.value == "stale" else "✕"
                line = f"  {status_icon} {entry.name}"
                if entry.description:
                    line += f" — {entry.description[:60]}"
                console.info(line)

    else:
        console.error(f"Unknown command: {command}. Type /help for available commands.")


if __name__ == "__main__":
    main()
