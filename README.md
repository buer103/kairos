<p align="center">
  <h1 align="center">καιρός — Kairos</h1>
  <p align="center"><strong>The right tool, at the right moment.</strong></p>
</p>

---

**Kairos** is an AI agent framework that combines the best ideas from
[Hermes](https://github.com/NousResearch/hermes-agent),
[DeerFlow](https://github.com/bytedance/deer-flow), and
[Claude Code](https://github.com/anthropics/claude-code).

Named after the ancient Greek word for *the decisive moment* — the
instant when an archer releases the bowstring. An agent should act not
just whenever it can, but exactly when it should.

## 🎯 Design Philosophy

> **Hermes** taught us that agents should learn from experience.  
> **DeerFlow** taught us that cross-cutting concerns deserve a pipeline.  
> **Claude Code** taught us that permission boundaries matter.

Kairos weaves these together into three core principles:

| Principle | What it means |
|-----------|--------------|
| **Learn at the right moment** | Skills that evolve with use, not one-shot prompts |
| **Intercept at the right stage** | A middleware pipeline that keeps the core loop clean |
| **Control at the right boundary** | Fine-grained permissions, not just a yolo switch |

## 🏗️ Architecture (planned)

```
User Message
    │
    ▼
┌──────────────────────┐
│   Middleware Pipeline │  ← DeerFlow-inspired hooks
│   (before/after/wrap) │
├──────────────────────┤
│   Agent Loop          │  ← Hermes-style while + tool calling
│   (think → act → repeat) │
├──────────────────────┤
│   Tool Registry       │  ← Self-registering tools
├──────────────────────┤
│   Model Providers     │  ← OpenAI-compatible abstraction
├──────────────────────┤
│   Memory & Skills     │  ← Persistent, evolving knowledge
│   + Curator           │     with automatic lifecycle management
└──────────────────────┘
    │
    ▼
Final Response
```

## 🚧 Status

**Pre-alpha — architecture design in progress.**

Currently studying the source code of Hermes, DeerFlow, and Claude Code
to extract the best patterns from each.

## 📖 The Name

*Kairos* (καιρός) vs *Chronos* (χρόνος):

- **Chronos** is clock time — sequential, quantitative, ticking forward.
- **Kairos** is the opportune moment — qualitative, decisive, the right
  instant to act.

An agent framework shouldn't just process messages in sequence.  
It should recognize *when* to act, *what* to call, and *how* to learn.

---

<p align="center">
  <sub>Built with ❤️ by <a href="https://github.com/buer103">buer103</a></sub>
</p>
