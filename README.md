<p align="center">
  <h1 align="center">καιρός — Kairos</h1>
  <p align="center"><strong>The right tool, at the right moment.</strong></p>
</p>

---

**Kairos** is an AI agent framework that inherits from
[Hermes](https://github.com/NousResearch/hermes-agent) and
[DeerFlow](https://github.com/bytedance/deer-flow),
adding four original capabilities neither provides.

Named after the ancient Greek word for *the decisive moment* —
the instant when an archer releases the bowstring.

## 🎯 Design

| From | What |
|------|------|
| **Hermes** | Agent Loop, Tool Registry, Chat CLI, Skills+Curator, Session Search, Gateway, RL Training, Model Providers |
| **DeerFlow** | Middleware Pipeline, Sub-Agent Factory, Sandbox, Typed State, Context Compression |
| **Kairos (new)** | RAG Engine, Structured Knowledge, Evidence Chain, Confidence+Citation, Customizable System Prompt |

## 🏗️ Architecture

```
User Message
    │
    ▼
┌──────────────────────────────────────────────┐
│              Middleware Pipeline              │
│  SkillLoader → EvidenceTracker → Compress    │
│  → ToolRateLimit → ConfidenceScorer          │
├──────────────────────────────────────────────┤
│              Agent Loop (ReAct)               │
│         think → tool_call → observe           │
├──────────────┬───────────────────────────────┤
│  Tool Registry│     Infrastructure           │
│  (self-reg)  │  RAG Engine · Knowledge Store │
│              │  Evidence DB · Vector Store   │
├──────────────┴───────────────────────────────┤
│  Model Providers · Session Search · Gateway  │
│  Skills+Curator · Sub-Agent · RL Training    │
└──────────────────────────────────────────────┘
```

## 🚀 Quick Start (coming soon)

```bash
pip install kairos

# Interactive chat
kairos chat

# Single query
kairos run "Diagnose the fault in log.txt"

# With your own tools
kairos chat --tools my_tools.py --knowledge my_schema.py
```

```python
from kairos import Agent

agent = Agent(
    tools=[my_log_tool, my_signal_tool],
    knowledge=MyDiagnosisSchema,
    middlewares=["evidence", "confidence"]
)

result = agent.run("Diagnose log-20260508.txt")
print(result.conclusion)   # "controller_overheat"
print(result.confidence)   # 0.92
print(result.evidence)     # [Step1, Step2, Step3]
```

## 📖 Full Architecture

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — 18 modules, full comparison
matrix against Hermes and DeerFlow, three-layer classification, and MVP breakdown.

## 🚧 Status

**Pre-alpha — implementing Phase 1 modules.**

- [x] Architecture design
- [ ] `core/` — Agent Loop, Middleware hooks, Typed State
- [ ] `providers/` — Model abstraction
- [ ] `tools/` — Self-registering registry
- [ ] `chat/` — CLI interface
- [ ] `prompt/` — System prompt engine
- [ ] `infra/` — RAG, Knowledge Store, Evidence DB
- [ ] `middleware/` — Evidence, Confidence, Compress
- [ ] `agents/` — Sub-Agent factory

---

<p align="center">
  <sub>Built by <a href="https://github.com/buer103">buer103</a></sub>
</p>
