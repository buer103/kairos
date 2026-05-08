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
        print("  kairos chat          Interactive chat mode")
        print("  kairos run <query>   Single query mode")
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
    else:
        print(f"Unknown command: {args[0]}")
        print("Usage: kairos chat | kairos run <query>")


def _chat_mode(args: list[str]):
    """Interactive chat loop."""
    from kairos import Agent
    from kairos.providers.base import ModelConfig

    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  No API key found. Set DEEPSEEK_API_KEY or OPENAI_API_KEY.")
        return

    model = ModelConfig(api_key=api_key)
    agent = Agent(model=model)

    print("🤖 Kairos chat (type /exit to quit)")
    print()

    while True:
        try:
            user_input = input("🤖 Kairos> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("/exit", "/quit"):
            print("👋 Goodbye!")
            break

        print()
        result = agent.run(user_input)
        print(f"🤖 {result['content']}")
        if result.get("confidence"):
            print(f"   📊 confidence: {result['confidence']:.2f}")
        print()


def _run_mode(args: list[str]):
    """Single query mode."""
    if not args:
        print("Usage: kairos run <query>")
        return

    from kairos import Agent
    from kairos.providers.base import ModelConfig

    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  No API key found.")
        return

    query = " ".join(args)
    model = ModelConfig(api_key=api_key)
    agent = Agent(model=model)

    result = agent.run(query)
    print(result["content"])


if __name__ == "__main__":
    main()
