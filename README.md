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
| **Hermes** | Agent Loop, Tool Registry, Skills+Curator, Session Search, Gateway (11 platforms), Cron, Delegation, RL Training, Model Providers |
| **DeerFlow** | Middleware Pipeline (18 layers), Sub-Agent Factory, Sandbox, Typed ThreadState, Context Compression (3 tiers) |
| **Kairos (new)** | RAG Engine, Structured Knowledge, Evidence Chain, Confidence+Citation, Credential Pool, Rich TUI, Plugin System, Trace ID, Tiered Memory, Provider Registry |

## 🏗️ Architecture

```
User Message → Gateway (11 platforms)
    │
    ▼
┌──────────────────────────────────────────────────────┐
│              Middleware Pipeline (18 layers)          │
│  ThreadData → Uploads → Dangling → SkillLoader       │
│  → Compress → Todo → Memory → ViewImage              │
│  → Evidence → ToolArgRepair → Confidence             │
│  → LLMRetry → Logging → SubagentLimit → Title        │
│  → MemoryMiddleware → Clarify                        │
│  + SecurityMiddleware (optional)                     │
├──────────────────────────────────────────────────────┤
│              Agent Loop (ReAct + Stateful)            │
│         think → tool_call → observe → repeat          │
├──────────────┬───────────────────────────────────────┤
│  Tool Registry│     Infrastructure                   │
│  (auto-reg)  │  RAG · Knowledge · Evidence DB        │
│  23 tools    │  Sandbox (local/docker/ssh)           │
│              │  Tiered Memory (profile/timeline/facts)│
├──────────────┴───────────────────────────────────────┤
│  Model Providers (7 profiles) · Memory · Skills      │
│  Cron · Session Search · Delegation · RL Training    │
│  Trace ID · Observability · Security · Plugins       │
└──────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

```bash
pip install kairos-agent

# Generate default config
kairos config init

# Set your API key (or add to ~/.config/kairos/config.yaml)
export DEEPSEEK_API_KEY=sk-...

# Interactive chat (10+ skins, tab completion, slash commands)
kairos chat

# Single query
kairos run "Explain the Kubernetes scheduler in 3 bullet points"

# List available providers
kairos config providers
```

```python
from kairos import Agent
from kairos.providers.base import ModelConfig
from kairos.providers.registry import ProviderRegistry

# Quick: any OpenAI-compatible API
agent = Agent(model=ModelConfig(api_key="sk-..."))
result = agent.run("What is 2+2?")
print(result["content"])  # "4"

# Use a named provider profile
registry = ProviderRegistry()
config = registry.make_config("deepseek", api_key="sk-...")
agent = Agent(model=config)

# Full pipeline with tiered memory, RAG, skills, and streaming
agent = Agent.build_default(
    model=ModelConfig(api_key="sk-..."),
    agent_name="DiagnosisBot",
    role_description="You diagnose system faults from logs.",
    rag_store=my_rag,
    enable_tiered_memory=True,
)
result = agent.run("Diagnose log-20260508.txt")
print(result["content"])
print(result["confidence"])  # 0.92
print(result["trace_context"])  # TraceContext for full-chain observability
```

## 📦 Modules (62)

| Layer | Modules |
|-------|---------|
| **Core** | Agent Loop (ReAct + Stateful), 18-layer Middleware Pipeline, Typed ThreadState, Trace ID |
| **Tools** | 23 built-in tools (file, terminal, web, browser, MCP, vision, rag, knowledge, delegate) |
| **Providers** | 7 profiles (DeepSeek, OpenRouter, Groq, Qwen, OpenAI, Anthropic, Gemini) + Credential Pool |
| **Memory** | 3-tier memory (Profile/Timeline/Facts), confidence≥0.7 filter, per-agent isolation |
| **Infra** | RAG Engine, Structured Knowledge, Evidence DB, Vector Store |
| **Skills** | Curator lifecycle (install/update/remove), semantic retrieval, marketplace |
| **Gateway** | 11 platform adapters + signatures + readiness + pairing + rate limiting |
| **Cron** | SQLite-backed scheduler (cron expression, repeat, pause/resume) |
| **Delegation** | Sub-agent spawning (ThreadPoolExecutor, parallel batch, orchestrator) |
| **Sandbox** | 3 execution backends (local/Docker/SSH), sandbox middleware |
| **Security** | 6-layer security (file safety, URL safety, path security, content redaction, guardrails) |
| **Observability** | Error classifier, usage tracker, agent insights, health endpoints |
| **Training** | Trajectory recorder (ShareGPT JSONL), RL environment + 4 reward functions |
| **CLI** | Rich TUI (10 skins, tab completion, slash commands, setup wizard) |
| **Config** | Pydantic schema validation, YAML/JSON config + env var fallback |
| **Plugins** | Plugin manifest + Manager (load/unload/reload), 3 built-in plugins |
| **Deploy** | Docker multi-stage build + docker-compose + HEALTHCHECK |
| **Gateway** | Graceful shutdown, session drain, webhook signature verification (6 platforms) |

## ✅ Status

**Beta — 695 tests passing. 46 commits. v0.15.0.**

- [x] Architecture design + 4-framework comparison matrix
- [x] Phase 1: Agent Loop, Prompt Engine, RAG, Knowledge, Evidence, Middleware
- [x] Phase 2: Memory, Skills, Session Search, Sandbox
- [x] Phase 3: Gateway (4 platforms), Training (RL recorder + env)
- [x] Phase 4: Middleware parity with DeerFlow (5 → 18 layers)
- [x] Phase 5: LLM error handling, credential pool, SSE streaming, interrupt/resume
- [x] Phase 6: Cron scheduler, Rich TUI, Sandbox wiring, Sub-agent delegation
- [x] Phase 7: Gateway expansion (7 → 11 platforms), layered context compression
- [x] Phase 8: Config system, pyproject.toml, CLI polish
- [x] Phase 9: Full test coverage (299→566 tests)
- [x] Phase 10: Security layer + Observability + Sub-agent Orchestrator
- [x] Phase 11: Browser/MCP/Vision tools + Tiered Memory + Streaming v2
- [x] Phase 12: Docker deployment + Graceful shutdown + Webhook signatures
- [x] Phase 13: Trace ID full-chain + Memory 3-tier + Provider Registry (566→695)

### Next up

- [ ] Phase 14: Batch runner, Interactive permissions, ACP IDE integration
- [ ] Phase 15: Documentation site, tutorials, CI/CD pipeline
- [ ] Phase 16: v1.0 release

---

<p align="center">
  <sub>Built by <a href="https://github.com/buer103">buer103</a> · 46 commits · 695 tests · v0.15.0</sub>
</p>
