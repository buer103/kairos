<p align="center">
  <h1 align="center">καιρός — Kairos</h1>
  <p align="center"><strong>The right tool, at the right moment.</strong></p>
</p>

---

**Kairos** is an AI agent framework that inherits from
[Hermes](https://github.com/NousResearch/hermes-agent) and
[DeerFlow](https://github.com/bytedance/deer-flow),
adding original capabilities neither provides.

Named after the ancient Greek word for *the decisive moment* —
the instant when an archer releases the bowstring.

## 🎯 Design

| From | What |
|------|------|
| **Hermes** | Agent Loop, Tool Registry, Skills+Curator, Memory, Session Search, Gateway (11 platforms), Cron, Delegation, RL Training, Model Providers |
| **DeerFlow** | Middleware Pipeline (16 layers), Sub-Agent Factory, Sandbox, Typed ThreadState, Context Compression |
| **Kairos (new)** | RAG Engine, Structured Knowledge, Evidence Chain, Confidence+Citation, Plugin System, Credential Pool, Rich TUI |

## 🏗️ Architecture

```
User Message → Gateway (11 platforms)
    │
    ▼
┌──────────────────────────────────────────────────────┐
│              Middleware Pipeline (16 layers)          │
│  ThreadData → Uploads → Dangling → SkillLoader       │
│  → Compress → Todo → Memory → ViewImage              │
│  → Evidence → ToolArgRepair → Confidence             │
│  → LLMRetry → SubagentLimit → Title → Clarify        │
├──────────────────────────────────────────────────────┤
│              Agent Loop (ReAct + Stateful)            │
│         think → tool_call → observe → repeat          │
├──────────────┬───────────────────────────────────────┤
│  Tool Registry│     Infrastructure                   │
│  (auto-reg)  │  RAG · Knowledge · Evidence DB        │
│  22 tools    │  Sandbox (local/docker/ssh)           │
├──────────────┴───────────────────────────────────────┤
│  Model Providers · Memory · Skills · Cron            │
│  Session Search · Delegation · RL Training           │
└──────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

```bash
pip install kairos-agent

# Generate default config
kairos config init

# Set your API key (or add to ~/.config/kairos/config.yaml)
export DEEPSEEK_API_KEY=sk-...

# Interactive chat
kairos chat

# Single query
kairos run "Explain the Kubernetes scheduler in 3 bullet points"

# List available tools
kairos chat  →  /tools

# Manage cron jobs
kairos cron list
kairos cron add "daily-report" "0 9 * * *"
```

```python
from kairos import Agent
from kairos.providers.base import ModelConfig

# Minimal — just an API key
agent = Agent(model=ModelConfig(api_key="sk-..."))
result = agent.run("What is 2+2?")
print(result["content"])  # "4"

# Full pipeline with custom tools and knowledge
agent = Agent.build_default(
    model=ModelConfig(api_key="sk-..."),
    agent_name="DiagnosisBot",
    role_description="You diagnose system faults from logs.",
    rag_store=my_rag,
    skills_dir="~/.kairos/skills",
)
result = agent.run("Diagnose log-20260508.txt")
print(result["content"])
print(result["confidence"])  # 0.92
```

## 📦 Modules

| Layer | Modules |
|-------|---------|
| **Core** | Agent Loop (ReAct + Stateful), 16-layer Middleware Pipeline, Typed ThreadState |
| **Tools** | 22 built-in tools (file, terminal, web, rag, knowledge, cron, delegate) |
| **Providers** | OpenAI-compatible + Credential Pool (multi-key rotation + retry) |
| **Infra** | RAG Engine, Structured Knowledge, Evidence DB, Vector Store |
| **Memory** | Persistent memory (SQLite), auto-injection middleware |
| **Skills** | Curator lifecycle (install/update/remove), skill loader middleware |
| **Gateway** | 11 platform adapters (CLI, Telegram, WeChat, Slack, Discord, Feishu, WhatsApp, Signal, Line, Matrix, IRC) |
| **Cron** | SQLite-backed scheduler (cron expression, repeat, pause/resume) |
| **Delegation** | Sub-agent spawning (ThreadPoolExecutor, timeout, batch) |
| **Sandbox** | 3 execution backends (local/Docker/SSH), sandbox middleware |
| **Training** | Trajectory recorder (ShareGPT), RL environment + rewards |
| **CLI** | Rich TUI (4 skins, 10+ slash commands, spinner, panels) |
| **Config** | YAML/JSON config + env var fallback (`kairos config init`) |
| **Plugins** | Plugin manifest + Manager (load/unload/reload) |

## ✅ Status

**Alpha — 566 tests passing.**

- [x] Architecture design + module comparison matrix
- [x] Phase 1: Agent Loop, Prompt Engine, RAG, Knowledge, Evidence, Middleware
- [x] Phase 2: Memory, Skills, Session Search, Sandbox
- [x] Phase 3: Gateway (4 platforms), Training (RL recorder + env)
- [x] Phase 4: Middleware parity with DeerFlow (5 → 16 layers)
- [x] Phase 5: LLM error handling, credential pool, SSE streaming, interrupt/resume
- [x] Phase 6: Cron scheduler, Rich TUI, Sandbox wiring, Sub-agent delegation
- [x] Phase 7: Gateway expansion (7 → 11 platforms), layered context compression
- [x] Phase 8: Config system, pyproject.toml, CLI polish, README
- [x] Phase 9: Full test coverage — 7 new test files, 299→566 tests (+89%), 21 modules covered

---

<p align="center">
  <sub>Built by <a href="https://github.com/buer103">buer103</a> · 36 commits · 566 tests</sub>
</p>
