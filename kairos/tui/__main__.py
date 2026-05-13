"""Kairos TUI — Textual-based terminal user interface.

Usage:
    kairos tui                      Launch TUI with defaults
    kairos tui --skin hacker        Use hacker skin
    kairos tui --verbose            Verbose tool output
    kairos tui --resume <name>      Resume a saved session

Entry point for python -m kairos.tui
"""

from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None) -> None:
    """Main entry point for Kairos TUI."""
    parser = argparse.ArgumentParser(
        prog="kairos tui",
        description="Kairos Textual TUI — interactive agent chat",
    )
    parser.add_argument("--skin", default="default",
                        choices=["default", "hacker", "retro", "minimal"])
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed tool output")
    parser.add_argument("--resume", type=str,
                        help="Resume a saved session by name")
    parser.add_argument("--model", type=str,
                        help="Model name override")
    parser.add_argument("--base-url", type=str,
                        help="API base URL override")
    args = parser.parse_args(argv)

    # ── Build agent ─────────────────────────────────────────────
    api_key = (
        os.environ.get("KAIROS_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        print("❌ No API key found. Set KAIROS_API_KEY or DEEPSEEK_API_KEY.")
        sys.exit(1)

    base_url = args.base_url or os.environ.get(
        "KAIROS_BASE_URL", "https://api.deepseek.com/v1"
    )
    model_name = args.model or os.environ.get(
        "KAIROS_MODEL", "deepseek-chat"
    )

    from kairos.providers.base import ModelConfig
    from kairos.core.stateful_agent import StatefulAgent

    model = ModelConfig(api_key=api_key, base_url=base_url, model=model_name)
    agent = StatefulAgent(model=model)

    # Resume session if requested
    if args.resume:
        if agent.load_session(args.resume):
            print(f"📂 Resumed session: {args.resume}")
        else:
            print(f"❌ Session not found: {args.resume}")
            sys.exit(1)

    # ── Launch TUI ──────────────────────────────────────────────
    from kairos.tui.app import KairosTUI

    app = KairosTUI(
        agent=agent,
        skin=args.skin,
        verbose=args.verbose,
    )
    app.run()


if __name__ == "__main__":
    main()
