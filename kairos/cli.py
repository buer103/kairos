"""CLI entry point for Kairos."""

from __future__ import annotations

import os
import sys

def main():
    """Main entry point: kairos chat / kairos run."""
    args = sys.argv[1:]

    if not args:
        print("Kairos — The right tool, at the right moment.")
        print()
        print("Usage:")
        print("  kairos chat          Interactive chat mode (Rich TUI)")
        print("  kairos run <query>   Single query mode")
        print("  kairos cron          Cron scheduler management")
        print("  kairos --version     Show version")
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
    else:
        print(f"Unknown command: {args[0]}")
        print("Usage: kairos chat | kairos run <query> | kairos cron")


def _chat_mode(args: list[str]):
    """Interactive chat loop with Rich TUI."""
    from kairos import Agent
    from kairos.providers.base import ModelConfig
    from kairos.cli.rich_ui import KairosConsole

    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        console = KairosConsole()
        console.error("No API key found. Set DEEPSEEK_API_KEY or OPENAI_API_KEY.")
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

        # ── Slash commands ──────────────────────────────────
        if user_input.startswith("/"):
            _handle_slash(console, user_input, agent, model)
            continue

        # ── Normal query ────────────────────────────────────
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

        # Show tool calls if verbose
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
    from kairos.cli.rich_ui import KairosConsole

    if not args:
        console = KairosConsole()
        console.error("Usage: kairos run <query>")
        return

    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
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
    from kairos.cli.rich_ui import KairosConsole
    from kairos.cron.scheduler import CronScheduler, Job, CronSchedule, JobStatus

    console = KairosConsole()
    scheduler = CronScheduler()

    if not args or args[0] == "list":
        jobs = scheduler.list()
        console.show_cron_jobs(jobs)
    elif args[0] == "add":
        if len(args) < 3:
            console.error("Usage: kairos cron add <name> <cron_expr> [command...]")
            return
        # Parse cron expression (min hour dom month dow)
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
        job = scheduler.pause(args[1])
        if job:
            console.success(f"Paused job: {job.name}")
        else:
            console.error(f"Job not found: {args[1]}")
    elif args[0] == "resume" and len(args) >= 2:
        job = scheduler.resume(args[1])
        if job:
            console.success(f"Resumed job: {job.name}")
        else:
            console.error(f"Job not found: {args[1]}")
    elif args[0] == "cancel" and len(args) >= 2:
        job = scheduler.cancel(args[1])
        if job:
            console.success(f"Cancelled job: {job.name}")
        else:
            console.error(f"Job not found: {args[1]}")
    elif args[0] == "remove" and len(args) >= 2:
        scheduler.remove(args[1])
        console.success(f"Removed job: {args[1]}")
    else:
        console.error(f"Unknown cron command: {' '.join(args)}")
        console.info("Usage: kairos cron [list|add|pause|resume|cancel|remove]")


def _parse_field(field: str) -> list[int]:
    """Parse a cron field like '*' or '1,2,3' or '*/5'."""
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
        status = "ON" if console.verbose else "OFF"
        console.info(f"Verbose tool output: {status}")

    elif command == "/skin":
        if len(parts) >= 2:
            console.set_skin(parts[1])
        else:
            console.info(f"Current skin: {console.skin_name}. Available: default, hacker, retro, minimal")

    elif command == "/tools":
        console.show_tools()

    elif command == "/cron":
        from kairos.cron.scheduler import CronScheduler
        scheduler = CronScheduler()
        if len(parts) >= 2 and parts[1] == "list":
            jobs = scheduler.list()
            console.show_cron_jobs(jobs)
        else:
            jobs = scheduler.list()
            console.show_cron_jobs(jobs)

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

    else:
        console.error(f"Unknown command: {command}. Type /help for available commands.")


if __name__ == "__main__":
    main()
