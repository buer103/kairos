# Kairos — Framework Architecture (v0.15.0-dev)

> **Kairos** (καιρός): the decisive moment.  
> An agent should act not whenever it can, but exactly when it should.

---

## 1. Identity

Kairos is an AI agent framework that fuses **Hermes** and **DeerFlow**,
adding original capabilities neither provides. It serves three roles:

| Role | How | Example |
|------|-----|---------|
| **Standalone CLI** | `kairos chat` / `kairos run` | Like `hermes` in the terminal |
| **Python Library** | `from kairos import Agent` | Embedded in your projects |
| **Business Platform** | Build your agent on Kairos | Diagnosis, legal research, code review |

### Design Lineage

| From | Modules Adopted |
|------|----------------|
| **Hermes** | Agent Loop, Tool Registry, Skills+Curator, Session Search, Gateway, RL Training, Model Providers, Cron |
| **DeerFlow** | Middleware Pipeline, Sub-Agent Factory, Sandbox, Typed State, Context Compression |
| **Kairos (new)** | RAG Engine, Structured Knowledge, Evidence Chain, Confidence+Citation, Credential Pool, Plugin System, Rich TUI, Trace ID, Tiered Memory, Provider Registry, Security Layer, Observability |

---

## 2. Module Map

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           Kairos Framework (62 modules)                    │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │  Agent   │  │Middleware│  │  Tool    │  │   Chat   │  │  System  │  │
│  │  Loop    │  │ Pipeline │  │ Registry │  │   CLI    │  │  Prompt  │  │
│  │ (Hermes) │  │(DeerFlow)│  │ (Hermes) │  │ (Hermes) │  │ (Kairos) │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
│                                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │   RAG    │  │Knowledge │  │ Evidence │  │Confidence│  │  Skills  │  │
│  │  Engine  │  │  Store   │  │  Chain   │  │ +Citation│  │ +Curator │  │
│  │ (Kairos) │  │ (Kairos) │  │ (Kairos) │  │ (Kairos) │  │ (Hermes) │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
│                                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │  Model   │  │ Sub-Agent│  │  Typed   │  │ Sandbox  │  │ Context  │  │
│  │Providers │  │ Factory  │  │  State   │  │(DeerFlow)│  │ Compress │  │
│  │ (Hermes) │  │(DeerFlow)│  │(DeerFlow)│  │          │  │(DeerFlow)│  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
│                                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │ Session  │  │ Gateway  │  │   RL     │  │  Tiered  │               │
│  │  Search  │  │  11平台   │  │ Training │  │  Memory  │               │
│  │ (Hermes) │  │ (Hermes) │  │ (Hermes) │  │ (Kairos) │               │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘               │
│                                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │  Trace   │  │ Security │  │Observab- │  │ Provider │               │
│  │    ID    │  │  Layer   │  │  ility   │  │ Registry │               │
│  │ (Kairos) │  │ (Kairos) │  │ (Kairos) │  │ (Kairos) │               │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘               │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Three-Layer Classification

Every capability belongs to exactly one of these layers:

| Layer | Rule | What goes here |
|-------|------|---------------|
| **Tool** | Agent actively decides when to invoke | `rag_search`, `web_search`, `browser`, `MCP`, `vision` |
| **Middleware** | Framework auto-executes at lifecycle hooks | EvidenceTracker, ConfidenceScorer, ContextCompress, SkillLoader |
| **Infrastructure** | Shared storage/engine underneath | VectorStore, KnowledgeStore, EvidenceDB, MemoryBackend |

---

## 3. Middleware Pipeline (18 layers)

Kairos has a production-grade 18-layer middleware pipeline, exceeding DeerFlow's original 11:

| # | Layer | Hook | Source |
|---|-------|------|:--:|
| 1 | ThreadData | before_agent | DeerFlow |
| 2 | Uploads | before_agent | DeerFlow |
| 3 | DanglingToolCall | wrap_model_call | DeerFlow |
| 4 | SkillLoader | before_agent | Hermes |
| 5 | ContextCompressor | before_model | DeerFlow |
| 6 | Todo | before_model | DeerFlow |
| 7 | Memory (inject) | before_model | Kairos |
| 8 | ViewImage | before_model | DeerFlow |
| 9 | EvidenceTracker | wrap_tool_call | Kairos |
| 10 | ToolArgRepair | wrap_tool_call | Kairos |
| 11 | ConfidenceScorer | after_model | Kairos |
| 12 | LLMRetry | wrap_model_call | Kairos |
| 13 | Logging | wrap_model_call | Kairos |
| 14 | SubagentLimit | after_model | DeerFlow |
| 15 | Title | after_model | DeerFlow |
| 16 | Memory (write) | after_agent | Kairos |
| 17 | Clarification | wrap_tool_call | DeerFlow |
| *opt* | SecurityMiddleware | before/after_model | Kairos |
| *opt* | SandboxMiddleware | wrap_tool_call | Kairos |

### 6 Hook Types

| Hook | When it fires | Purpose |
|------|--------------|---------|
| `before_agent` | Once, before any LLM call | Initialize state, load skills |
| `after_agent` | Once, after agent finishes | Score confidence, commit memory |
| `before_model` | Before every LLM call | Inject context, compress, load reminders |
| `after_model` | After every LLM call | Validate output, extract facts |
| `wrap_model_call` | Wraps the LLM call | Fix dangling calls, retry on error |
| `wrap_tool_call` | Wraps each tool execution | Record evidence, sandbox, security |

---

## 4. Tool Ecosystem (23 tools)

| Category | Tools |
|----------|-------|
| **File** | `read_file`, `write_file`, `list_files` |
| **Terminal** | `terminal` |
| **Web** | `web_search`, `web_fetch`, `web_scrape`, `web_screenshot`, `web_search_advanced`, `web_form_submit` |
| **Knowledge** | `rag_search`, `knowledge_lookup` |
| **Skills** | `skills_list`, `skill_view`, `skill_manage` |
| **Vision** | `vision_analyze`, `vision_compare`, `vision_screenshot_analyze` |
| **MCP** | `mcp_connect`, `mcp_call_tool`, `mcp_list_servers`, `mcp_disconnect` |
| **Delegation** | `delegate_task` |

---

## 5. Provider Registry (7 profiles)

| Provider | Type | Default Model |
|----------|------|:--:|
| OpenAI | OpenAI-compat | `gpt-4o` |
| DeepSeek | OpenAI-compat | `deepseek-chat` |
| OpenRouter | OpenAI-compat | `openai/gpt-4o` |
| Groq | OpenAI-compat | `llama-3.3-70b-versatile` |
| Qwen (通义千问) | OpenAI-compat | `qwen-plus` |
| Anthropic | Native SDK | `claude-sonnet-4-20250514` |
| Gemini | Native SDK | `gemini-2.5-flash` |

Plus Credential Pool with multi-key rotation, 429 handling, and exponential backoff.

---

## 6. Tiered Memory (3 tiers)

DeerFlow-compatible 3-tier memory with confidence filtering:

| Tier | Semantics | Confidence |
|------|----------|:--:|
| **Profile** (画像) | Stable user attributes, overwrite | 1.0 (always) |
| **Timeline** (时间线) | Chronological events, append-only | 1.0 (always) |
| **Facts** | Extracted knowledge with confidence | ≥ 0.7 (filtered) |

Features: per-agent isolation, max injection token budget (2000), TTL expiry, 5 fact categories.

---

## 7. Trace ID (Full-Chain Observability)

Every `agent.run()` creates a span. Sub-agents create child spans forming a tree:

```
trace-abc123
├── root_2a9a (depth=0) — parent agent
│   ├── sub_d216 (depth=1) — sub-agent 1
│   │   └── sub_ghi7 (depth=2) — sub-sub-agent
│   └── sub_jkl0 (depth=1) — sub-agent 2
```

Features: span tree reconstruction, JSONL persistence, context-var based implicit propagation, `get_trace()` / `list_traces()` API.

---

## 8. Gateway (11 platforms)

| Platform | Features |
|----------|----------|
| **Telegram** | getUpdates polling, webhook, native media |
| **WeChat** | XML message parsing, signature verification, passive reply |
| **Slack** | Events API, Block Kit, HMAC verification |
| **Discord** | Gateway intents, Ed25519 signature |
| **Feishu** | Event subscription, HMAC-SHA256 |
| **WhatsApp** | Business API, HMAC-SHA256 |
| **Signal** | signal-cli integration |
| **Line** | Messaging API, base64-HMAC |
| **Matrix** | Client-Server API, keyed-HMAC |
| **IRC** | RFC 1459 compliant |
| **CLI** | Rich TUI, 10 skins, tab completion |

Plus: GatewayManager lifecycle, WebhookServer (7-platform), PairingManager (QR/OAuth), RateLimiter (sliding window), graceful shutdown.

---

## 9. Complete Module List (62 modules)

```python
kairos/
├── core/           # Agent loop, middleware, state, tracing
│   ├── loop.py          # Agent + build_default factory
│   ├── middleware.py     # 6 hook types + pipeline orchestrator
│   ├── state.py          # ThreadState, Case, Step
│   ├── stateful_agent.py # Multi-turn with session persistence
│   ├── thread_state.py   # Thread data management
│   ├── paths.py          # Path resolution
│   └── tracing.py        # TraceContext + TraceRecorder
├── middleware/      # 17 middleware layers + 2 optional
│   ├── clarify.py        # Clarification (must be last)
│   ├── compress.py       # Context v3 (trajectory + importance)
│   ├── confidence.py     # 6-factor weighted scoring
│   ├── dangling.py       # Tool call fix
│   ├── evidence.py       # Evidence step tracking
│   ├── llm_retry.py      # Retry + backoff + tool repair
│   ├── logging_mw.py     # Structured logging
│   ├── importance_scorer.py  # Per-message retention scoring
│   ├── trajectory_compressor.py # LLM-based summarization
│   ├── memory_mw.py      # Memory injection
│   ├── sandbox_mw.py     # Sandbox routing
│   ├── security_mw.py    # Input/output/tool guardrails
│   ├── skill_loader.py   # Skill injection
│   ├── subagent_limit.py # Concurrency cap
│   ├── thread_data.py    # Workspace management
│   ├── title.py          # Session title generation
│   ├── todo.py           # Todo list persistence
│   ├── uploads.py        # File upload handling
│   └── view_image.py     # Vision model injection
├── tools/           # 23 built-in tools
│   ├── registry.py       # Self-registering tool system
│   ├── builtin.py        # File/terminal/web tools
│   ├── browser_tools.py  # Advanced web interaction
│   ├── rag_search.py     # RAG retrieval
│   ├── knowledge_lookup.py  # Structured knowledge query
│   ├── skills_tool.py    # Skill lifecycle tools
│   ├── vision_tools.py   # Image analysis
│   └── mcp_tools.py      # MCP protocol client
├── providers/       # Model providers
│   ├── base.py           # ModelProvider (OpenAI-compat)
│   ├── registry.py       # 7 provider profiles
│   ├── credential.py     # Multi-key rotation pool
│   ├── anthropic_adapter.py # Claude native SDK
│   └── gemini_adapter.py    # Gemini native SDK
├── agents/          # Sub-agent system
│   ├── factory.py        # Type registration
│   ├── executor.py       # run_sync / run_parallel
│   ├── delegate.py       # DelegateTask tool + DelegationManager
│   ├── orchestrator.py   # Recursive delegation tree
│   └── types.py          # SubAgentType definitions
├── memory/          # Tiered memory
│   ├── store.py          # MemoryStore (legacy)
│   ├── backends.py       # SQLiteBackend / DictBackend
│   ├── middleware.py      # MemoryMiddleware (legacy)
│   ├── middleware_v2.py   # MemoryMiddlewareV2 (tiered)
│   └── tiers.py          # TieredMemoryStore (profile/timeline/facts)
├── gateway/         # 11 platform adapters
│   ├── server.py          # HTTP + SSE + /health /ready
│   ├── webhook.py         # Multi-platform webhook server
│   ├── signatures.py      # Ed25519 + HMAC verification
│   ├── manager.py         # Lifecycle + routing
│   ├── pairing.py         # QR/OAuth device flow
│   ├── protocol.py        # Unified message protocol
│   ├── ratelimit.py       # Sliding window rate limiter
│   └── adapters/          # 11 platform-specific adapters
├── infra/           # Infrastructure
│   ├── rag/               # VectorStore + adapters
│   ├── knowledge/         # KnowledgeSchema + Store
│   └── evidence/          # EvidenceDB tracker
├── security/        # 6-layer security
│   ├── file_safety.py     # Dangerous file detection
│   ├── url_safety.py      # Internal URL blocking
│   ├── path_security.py   # Path traversal prevention
│   ├── content_redact.py  # API key redaction
│   └── guardrails.py      # Input/output/tool guardrails
├── observability/   # Metrics + insights
│   ├── error_classifier.py # Sliding-window error detection
│   ├── usage_tracker.py    # Token/cost tracking
│   └── insights.py         # Health reports + anomaly detection
├── plugins/         # Plugin system
│   ├── builtin/           # 3 built-in plugins
│   │   ├── memory_plugin.py
│   │   ├── context_plugin.py
│   │   └── provider_plugin.py
│   └── ...                # Plugin manager
├── skills/          # Skill system
│   ├── manager.py         # Skill lifecycle
│   ├── marketplace.py     # install from GitHub/HF/URL
│   └── builtin/           # code-review, testing, debugging, github
├── training/        # RL training
│   ├── recorder.py        # ShareGPT JSONL trajectory
│   └── env.py             # RL environments + rewards
├── cron/            # Scheduled jobs
│   └── scheduler.py       # SQLite-backed cron engine
├── cli/             # Rich terminal UI
│   ├── rich_ui.py         # 10 skins + spinner + panels
│   ├── skin_engine.py     # YAML skin customization
│   ├── completions.py     # Tab completion engine
│   └── wizard.py          # Interactive setup wizard
├── chat/            # Session management
│   └── session.py         # Session persistence
├── config.py        # YAML/JSON config system
├── config_schema.py # Pydantic validation (10 sections)
├── logging.py       # Structured JSON logger
└── prompt/          # System prompt
    ├── template.py        # Modular template engine
    └── defaults.py        # Default templates
```

---

## 10. Comparison Matrix (Full Feature Set)

| # | Capability | Hermes | DeerFlow | Kairos | Source |
|---|-----------|:--:|:--:|:--:|--------|
| 1 | Agent Loop | ✅ while | ✅ LangGraph | ✅ while + streaming | Hermes |
| 2 | Middleware Pipeline | ❌ | ✅ 11-layer | ✅ 18-layer | DeerFlow + Kairos |
| 3 | Tool Registry | ✅ 50+ | ✅ 4-source | ✅ 23 tools | Hermes |
| 4 | Chat CLI | ✅ Ink TUI | ❌ Web only | ✅ Rich TUI (10 skins) | Hermes + Kairos |
| 5 | System Prompt | ✅ | ✅ modular | ✅ modular + hooks | Kairos |
| 6 | Sub-Agent Factory | 🟡 delegate | ✅ typed | ✅ typed + orchestrator | DeerFlow + Kairos |
| 7 | RAG Engine | ❌ | ❌ | ✅ vector + adapters | Kairos |
| 8 | Structured Knowledge | ❌ freeform | 🟡 5 categories | ✅ user-defined schemas | Kairos |
| 9 | Evidence Chain | ❌ | ❌ | ✅ Case→Step→Evidence | Kairos |
| 10 | Confidence + Citation | ❌ | ❌ | ✅ 6-factor scoring | Kairos |
| 11 | Skills + Curator | ✅ | ❌ | ✅ + semantic + marketplace | Hermes + Kairos |
| 12 | Model Providers | ✅ 20+ plugin | ✅ LiteLLM | ✅ 7 profiles + registry | Kairos |
| 13 | Credential Pool | ✅ | ❌ | ✅ multi-key + 429+backoff | Hermes + Kairos |
| 14 | Session Search | ✅ FTS5 | ❌ | ✅ FTS5 | Hermes |
| 15 | Gateway | ✅ 15+ platforms | ❌ | ✅ 11 platforms production | Hermes + Kairos |
| 16 | RL Training | ✅ Atropos | ❌ | ✅ ShareGPT + 4 rewards | Hermes |
| 17 | Typed State | ❌ | ✅ ThreadState | ✅ ThreadState | DeerFlow |
| 18 | Sandbox | ❌ | ✅ multi-provider | ✅ Local/Docker/SSH | DeerFlow |
| 19 | Context Compression | ✅ aggressive | ✅ LangChain | ✅ 3-tier (trajectory+importance) | Both + Kairos |
| 20 | Cron Scheduler | ✅ | ❌ | ✅ SQLite-backed | Hermes |
| 21 | Plugin System | ✅ | ❌ | ✅ 6 types + 3 built-in | Kairos |
| 22 | Security Layer | 🟡 basic | ❌ | ✅ 6-layer + guardrails | Kairos |
| 23 | Observability | ✅ plugin | ❌ | ✅ error+usage+insights | Kairos |
| 24 | Trace ID | ✅ | ✅ trace_id | ✅ span tree + JSONL | Kairos |
| 25 | Tiered Memory | ❌ flat | ✅ 3-tier | ✅ 3-tier + confidence filter | DeerFlow + Kairos |
| 26 | Docker Deploy | ✅ | ❌ | ✅ multi-stage + compose | Kairos |
| 27 | Graceful Shutdown | ❌ | ❌ | ✅ sessions+queue drain | Kairos |
| 28 | Webhook Signatures | ✅ | ❌ | ✅ 6-platform Ed25519+HMAC | Kairos |
| 29 | Config Schema | ❌ | ❌ | ✅ Pydantic 10-section validation | Kairos |
| 30 | Health Endpoints | ❌ | ❌ | ✅ /health /ready /detailed | Kairos |

---

## 11. Boundary: Framework vs Business

| Layer | Kairos (framework) provides | You (business) provide |
|-------|---------------------------|----------------------|
| Agent Loop | while loop + tool dispatch + streaming + interrupt | — |
| Middleware | 18 layers (evidence, confidence, compression, security, etc.) | Custom middleware for your domain |
| Tools | 23 generic tools (file, web, vision, MCP, knowledge) | Domain-specific tools |
| RAG | VectorStore, adapters, fusion engine | Knowledge base content |
| Knowledge | KnowledgeSchema base, KnowledgeStore | Your typed schemas |
| Evidence | Tracker, chain, replay | — |
| Memory | 3-tier store (profile/timeline/facts) | Memory content |
| Trace | Span tree, JSONL persistence | — |
| Providers | 7 profiles, credential pool | API keys |
| Gateway | 11 platform adapters, signatures, health | Platform credentials |
| Security | File/URL/path safety + guardrails | Custom rules |
| Observability | Error classifier, usage tracker, insights | — |

---

## 12. Version History

| Version | Highlights | Tests |
|---------|-----------|:--:|
| v0.10 | Agent Loop + Middleware + Tools + RAG + Knowledge | 299 |
| v0.11 | Security + Observability + Agent Loop v3 | — |
| v0.12 | Browser/MCP/Vision tools + Memory FTS5 + Orchestrator | — |
| v0.13 | Gateway deep upgrade (11 platforms production-grade) | — |
| v0.14 | CLI/TUI 10 skins + Plugin system + test boost (299→566) | 566 |
| v0.15-dev | Trace ID + Tiered Memory + Provider Registry + Docs | 695 |

---

<p align="center">
  <sub>Built by <a href="https://github.com/buer103">buer103</a> · 44 commits · 695 tests</sub>
</p>
