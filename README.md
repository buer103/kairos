<p align="center">
  <h1 align="center">καιρός — Kairos</h1>
  <p align="center"><strong>The right tool, at the right moment.</strong></p>
</p>

<p align="center">
  <a href="https://github.com/buer103/kairos/actions"><img src="https://github.com/buer103/kairos/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/kairos-agent"><img src="https://img.shields.io/pypi/v/kairos-agent" alt="PyPI"></a>
  <a href="https://github.com/buer103/kairos/blob/master/LICENSE"><img src="https://img.shields.io/github/license/buer103/kairos" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/pypi/pyversions/kairos-agent" alt="Python"></a>
</p>

---

**Kairos** is an AI agent framework that fuses the best of
[Hermes](https://github.com/NousResearch/hermes-agent) and
[DeerFlow](https://github.com/bytedance/deer-flow),
adding original capabilities neither provides.

Named after the ancient Greek word for *the decisive moment* —
the instant when an archer releases the bowstring.

## 🎯 Design

| From | What |
|------|------|
| **Hermes** | Agent Loop, Tool Registry, Skills+Curator, Session Search, Gateway, Cron, Delegation, RL Training, Providers |
| **DeerFlow** | Middleware Pipeline (20 layers), Sub-Agent Factory, Sandbox, Typed ThreadState, Context Compression (3 tiers) |
| **Kairos (new)** | RAG Engine, Structured Knowledge, Evidence Chain, Confidence+Citation, Credential Pool, Tiered Memory, Trace ID, Plugin System, Rich TUI |

## 🚀 Quick Start

```bash
pip install kairos-agent
export DEEPSEEK_API_KEY=sk-your-key
```

### CLI

```bash
# Interactive chat with live streaming
kairos chat

# One-shot query
kairos "Explain Kubernetes in 3 bullet points"

# List available commands
kairos --help
```

### As a Library

```python
from kairos import Agent
from kairos.providers.base import ModelConfig

# Simplest: any OpenAI-compatible API
agent = Agent(model=ModelConfig(api_key="sk-..."))
result = agent.run("What is 2+2?")
print(result["content"])   # "4"
print(result["confidence"])  # 0.95 (with ConfidenceScorer)

# Full pipeline with middleware, memory, and RAG
agent = Agent.build_default(
    model=ModelConfig(api_key="sk-..."),
    agent_name="DiagnosisBot",
    role_description="You diagnose system faults from logs.",
    rag_store=my_rag_store,
    memory_store=my_memory,
    enable_security=True,
)
result = agent.run("Diagnose log-20260508.txt")
```

### Custom Tools

```python
from kairos.tools.registry import register_tool

@register_tool(
    name="query_database",
    description="Run a SQL query against the company database",
    parameters={"sql": {"type": "string", "description": "SQL SELECT query"}},
)
def query_database(sql: str) -> str:
    return execute_sql(sql)  # your business logic
```

**More examples:** [`examples/`](examples/) — 7 runnable scripts covering basic usage, custom tools, multi-turn chat, RAG, middleware, Gateway, and FastAPI integration.

## 🏗️ Architecture

```
User Message → Gateway (11 platforms: Telegram, WeChat, Slack, ...)
    │
    ▼
┌──────────────────────────────────────────────────────┐
│           20-Layer Middleware Pipeline                │
│  ThreadData → Uploads → Dangling → SkillLoader       │
│  → ContextCompress(v3) → Todo → Memory → ViewImage   │
│  → Evidence → ToolArgRepair → Confidence             │
│  → LLMRetry (circuit breaker) → Logging              │
│  → SandboxAudit → LoopDetection → SubagentLimit      │
│  → Title → MemoryMiddleware → Clarify (last)         │
│  + SecurityMiddleware | TokenUsage (optional)         │
├──────────────────────────────────────────────────────┤
│           Agent Loop (ReAct + Stateful)               │
│       think → tool_call → observe → repeat            │
├──────────────┬───────────────────────────────────────┤
│  Tool Registry│     Infrastructure                   │
│  23 tools     │  RAG · Knowledge · Evidence DB        │
│  auto-register│  Sandbox (local/docker/ssh)           │
│  parallel-safe│  Tiered Memory (profile/timeline/facts)│
├──────────────┴───────────────────────────────────────┤
│  Providers (17 profiles) · CredentialPool · Failover │
│  Skills (Curator+Marketplace) · Session Search (FTS5)│
│  Delegation (Orchestrator) · Cron · RL Training      │
│  Trace ID · Observability · Security · Plugins       │
└──────────────────────────────────────────────────────┘
```

## 📦 Features

| Category | Capabilities |
|----------|-------------|
| **Core** | ReAct Agent Loop, StatefulAgent (multi-turn + persistence), Budget control, Interrupt/resume |
| **Middleware** | 20-layer pipeline (18 built-in + 2 optional), 6 lifecycle hooks, custom middleware support |
| **Tools** | 23 built-in (file, terminal, web, browser, MCP, vision, RAG, knowledge, delegate) + custom registration |
| **Providers** | 17 profiles (DeepSeek, OpenAI, Anthropic, Gemini, OpenRouter, Groq, Qwen…) + CredentialPool + Failover |
| **Memory** | 3-tier (Profile/Timeline/Facts), confidence≥0.7 filter, per-agent isolation, FTS5 search |
| **Gateway** | 11 platform adapters (Telegram, WeChat, Slack, Discord, Feishu, WhatsApp, Signal, Line, Matrix, IRC, CLI) |
| **Skills** | Curator lifecycle (active→stale→archived), Marketplace (GitHub/HF/URL/local), Self-improvement |
| **Security** | 3-level Permissions, SandboxAudit (dangerous command blocking), Guardrails, URL/File safety |
| **Observability** | ErrorClassifier, UsageTracker, AgentInsights, Prometheus /metrics endpoint |
| **Delegation** | Sub-agent spawning (ThreadPoolExecutor), Orchestrator, Policy (whitelist/blacklist) |
| **Training** | Trajectory recorder (ShareGPT JSONL), RL environment, 4 reward functions |
| **CLI** | Rich TUI (10 skins), tab completion, slash commands, streaming, setup wizard |
| **Deploy** | Docker multi-stage build, docker-compose, HEALTHCHECK, graceful shutdown |

## ✅ Status

**Beta — v0.16.0. 1,300+ tests passing. 74 commits. CI green (Python 3.10/3.11/3.12 + Docker).**

- [x] Phase 1–4: Agent Loop, Prompt Engine, RAG, Knowledge, Evidence, Middleware, Memory, Skills, Sandbox
- [x] Phase 5–8: Gateway (11 platforms), RL Training, Streaming, CredentialPool, Cron, Rich TUI, Config
- [x] Phase 9–12: Security, Observability, Sub-agent Orchestrator, Browser/MCP/Vision tools, Docker
- [x] Phase 13–14: Trace ID, Tiered Memory, Provider Registry, Parallel tools, Failover, Batch runner, Skill improver, Prompt caching, Sub-agent policy

### Roadmap

- [ ] Phase 15: ACP IDE integration, Interactive permissions, Documentation site, Tutorials
- [ ] Phase 16: v1.0 release

---

<p align="center">
  <sub>Built by <a href="https://github.com/buer103">buer103</a> · 74 commits · 1,300+ tests · v0.16.0</sub>
</p>
