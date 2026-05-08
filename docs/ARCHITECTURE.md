# Kairos — Framework Architecture (Complete)

> **Kairos** (καιρός): the decisive moment.  
> An agent should act not whenever it can, but exactly when it should.

---

## 1. Identity

Kairos is an AI agent framework that inherits from **Hermes** and **DeerFlow**,
adds four framework-level original capabilities, and serves three distinct roles.

### Three Roles

| Role | How | Example |
|------|-----|---------|
| **Standalone CLI** | `kairos chat` / `kairos run` | Like `hermes` in the terminal |
| **Python Library** | `from kairos import Agent` | Embedded in your projects |
| **Business Platform** | Build your agent on Kairos | Vehicle diagnosis, legal research, code review |

### Design Lineage

| From | Modules Adopted |
|------|----------------|
| **Hermes** | Agent Loop, Tool Registry, Chat CLI, Skills + Curator, Session Search, Gateway, RL Training, Model Providers |
| **DeerFlow** | Middleware Pipeline, Sub-Agent Factory, Sandbox, Typed State, Context Compression |
| **Kairos (new)** | RAG Engine, Structured Knowledge, Evidence Chain, Confidence + Citation, Customizable System Prompt |

---

## 2. Module Map

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           Kairos Framework                                │
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
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                              │
│  │ Session  │  │ Gateway  │  │   RL     │                              │
│  │  Search  │  │  多平台   │  │ Training │                              │
│  │ (Hermes) │  │ (Hermes) │  │ (Hermes) │                              │
│  └──────────┘  └──────────┘  └──────────┘                              │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Three-Layer Classification

Every capability belongs to exactly one of these layers:

| Layer | Rule | What goes here |
|-------|------|---------------|
| **Tool** | Agent actively decides when to invoke | `rag_search`, `knowledge_lookup`, domain tools |
| **Middleware** | Framework auto-executes at hooks | EvidenceTracker, ConfidenceScorer, ContextCompress, SkillLoader |
| **Infrastructure** | Shared storage/engine underneath | VectorStore, KnowledgeStore, EvidenceTrackerDB |

```
Agent Loop (pure — handles message ↔ LLM ↔ tool dispatch)

    ├── Tools (Agent-initiated)
    │   "I need to search → call rag_search()"
    │   "I need to look up a pattern → call knowledge_lookup()"
    │
    ├── Middleware (Framework-initiated, at hooks)
    │   wrap_tool_call  → EvidenceTracker records the step
    │   before_model    → ContextCompress checks token budget
    │   after_agent     → ConfidenceScorer evaluates output
    │
    └── Infrastructure (used by both tools and middleware)
        EvidenceTracker → EvidenceTrackerDB
        rag_search()    → VectorStore
        knowledge_lookup() → KnowledgeStore
```

---

## 3. Complete Module Specification

### 3.1 Agent Loop

| | Hermes | DeerFlow | Kairos |
|---|--------|----------|--------|
| **Mechanism** | Hand-written `while` loop | LangGraph `create_react_agent()` StateGraph | Hand-written `while` loop |
| **Purity** | Scattered with compression, memory, interrupt logic | Clean graph with middleware on nodes | Pure loop; all cross-cutting in middleware |

```
messages = [build_system_prompt(), user_message]
while iterations < max_iterations:
    if interrupted: break
    response = llm.chat(messages, tools)
    if response.tool_calls:
        for tc in response.tool_calls:
            result = execute_tool(tc.name, tc.args)
            messages.append(result)
    else:
        return response.content
```

The loop itself does **nothing** except message ↔ LLM ↔ tool dispatch.
Compression, evidence tracking, confidence scoring are all middleware.

### 3.2 Middleware Pipeline

6 hook types inherited from DeerFlow:

| Hook | When it fires | What it's for |
|------|--------------|---------------|
| `before_agent` | Once, before any LLM call | Initialize state, load skills |
| `after_agent` | Once, after agent finishes | Score confidence, commit memory |
| `before_model` | Before every LLM call | Inject context, compress, load reminders |
| `after_model` | After every LLM call | Validate output, truncate tool calls |
| `wrap_model_call` | Wraps the LLM call itself | Modify messages before sending, patch dangling calls |
| `wrap_tool_call` | Wraps each tool execution | Record evidence, rate-limit, intercept clarification |

MVP middleware stack (all optional):

```
SkillLoader          before_agent   → Semantic skill retrieval
EvidenceTracker      wrap_tool_call → Record every tool invocation as a step
ContextCompress      before_model   → Summarize when near token limit
ToolRateLimit        wrap_tool_call → Prevent runaway tool calls
ConfidenceScorer     after_agent    → Evaluate output confidence, attach evidence
```

### 3.3 Tool Registry

Self-registering, auto-discovered. A tool is a Python function + an OpenAI function-calling schema.

```python
@register_tool(
    name="rag_search",
    description="Search the knowledge base for relevant information",
    parameters={
        "query": {"type": "string", "description": "Search query"},
        "top_k": {"type": "integer", "description": "Number of results"}
    }
)
def rag_search(query: str, top_k: int = 5) -> dict:
    return vector_store.search(query, top_k)
```

Any `tools/*.py` file with `register_tool()` calls is auto-discovered.
No manual wiring. No LangChain dependency.

### 3.4 Chat CLI

Three interfaces, one Agent:

```bash
# Interactive chat (like hermes)
$ kairos chat
🤖 Kairos> 

# Interactive with business tools
$ kairos chat --tools my_tools.py --knowledge fault_schema.py

# Single query (non-interactive)
$ kairos run "诊断 log-20260508.txt 中的发动机故障"

# Resume a session
$ kairos chat --resume <session_id>
```

### 3.5 System Prompt

Kairos uses DeerFlow's **modular template** approach: the system prompt is a
template with named blocks, each independently overridable.

Built-in default template:
```
<role>
You are {agent_name}, an AI agent built on Kairos.
{role_description}
</role>

<personality>{soul}</personality>

<tools>{tools_section}</tools>
<knowledge>{knowledge_section}</knowledge>
<memory>{memory_section}</memory>

<evidence>
When evidence tracking is enabled, cite your reasoning steps.
</evidence>

<response>{response_style}</response>
```

Three levels of customization:

```python
# Level 1: Override specific blocks
agent = Agent(
    agent_name="AutoDiag",
    soul="You are a methodical vehicle diagnostics specialist...",
    response_style="Always output: 1) Root cause 2) Evidence 3) Confidence"
)

# Level 2: Replace the entire template
agent = Agent(
    system_template="""
    <role>You are {agent_name}, a {domain} expert.</role>
    <instructions>1. {first_step} 2. {second_step}</instructions>
    {tools_section}
    """
)

# Level 3: Post-render hook (programmatic modification)
agent = Agent(
    system_template="...",
    prompt_hook=lambda prompt: prompt + "\nAlways verify with knowledge base."
)
```

Template variables come from two sources:
- **Framework auto-fill**: `{tools_section}`, `{knowledge_section}`, `{memory_section}`
- **User-provided**: `{agent_name}`, `{soul}`, `{role_description}`, `{response_style}`, plus arbitrary custom variables

### 3.6 Sub-Agent Factory

Kairos adopts DeerFlow's typed factory model: Lead Agent creates typed
Sub-Agents via `task()` tool. Each Sub-Agent type has its own tool
allowlist/blocklist, max turns, timeout, and sandbox config.

| | Hermes delegate_task | DeerFlow task | Kairos task |
|---|------|------|------|
| **Types** | leaf / orchestrator | general-purpose / bash | User-definable per domain |
| **Tool control** | Toolsets | Blacklist or whitelist | Per-type blacklist / whitelist |
| **Concurrency** | Configurable | Hard capped [2,4] | Configurable + hard cap |
| **Timeout** | None | 900s | Configurable |
| **Context** | Isolated | Isolated | Isolated |
| **Sandbox** | Independent | Shared (lazy_init) | Configurable |
| **Evidence** | No | No | ✅ Sub-chain → parent |

```
Lead Agent
    │
    ├── task("Analyze segment 01", type="diagnose")
    │   └── EvidenceChain(sub_01)
    │
    ├── task("Analyze segment 02", type="diagnose")
    │   └── EvidenceChain(sub_02)
    │
    └── task("Cross-reference with repair manual", type="research")
        └── EvidenceChain(sub_03)

Parent EvidenceChain:
├── sub_01: found anomaly_A ← cited in final conclusion
├── sub_02: no anomaly
└── sub_03: matched pattern P-042 ← cited in final conclusion
```

### 3.7 RAG Engine

A framework-level retrieval-augmented generation pipeline. The Agent calls
`rag_search()` as a tool; the underlying infrastructure handles vector search,
multi-source fusion, and adapter routing.

| | Hermes | DeerFlow | Kairos |
|---|--------|----------|--------|
| **Vector search** | No | No | ✅ ChromaDB / FAISS |
| **External knowledge** | Not supported | Not supported | ✅ Pluggable adapters (Markdown, PDF, API) |
| **Skill retrieval** | By name only | N/A | ✅ Semantic skill matching |
| **Multi-source** | No | No | ✅ External KB + skills + session history |

```
RAG Engine
├── VectorStore          # Embedding + similarity search
├── Adapters             # Knowledge source connectors
│   ├── MarkdownAdapter  # Local .md files
│   ├── PDFAdapter       # PDF documents
│   └── APIAdapter       # External REST knowledge bases
├── SkillsIndex          # Semantic skill retrieval
└── Fusion               # Merge + deduplicate + rank across sources
```

### 3.8 Structured Knowledge Framework

User-defines typed schemas. Framework provides storage with structured queries.

| | Hermes | DeerFlow | Kairos |
|---|--------|----------|--------|
| **Format** | Freeform Markdown | 5 fixed categories | User-defined typed schemas |
| **Query** | grep by filename | Limited key-value on facts | Full structured query by any field |

```python
# Framework provides the base
class KnowledgeSchema(BaseModel):
    id: str
    created_at: datetime
    updated_at: datetime

# Business defines the schema
class FaultDiagnosis(KnowledgeSchema):
    signal_name: str
    log_pattern: str
    root_cause: str
    solution: str
    confidence: float

# Framework provides the store
store = KnowledgeStore(schema=FaultDiagnosis)
store.insert(FaultDiagnosis(signal_name="engine_temp", ...))
results = store.query({"root_cause": "controller_overheat"})
```

### 3.9 Evidence Chain

A middleware that records every tool invocation as a structured step,
building a traceable, replayable evidence chain for every agent session.

| | Hermes | DeerFlow | Kairos |
|---|--------|----------|--------|
| **Tracking** | Flat message transcript | LangGraph state snapshots | Structured Case → Step → Evidence |
| **Granularity** | Per message | Per graph node execution | Per tool invocation |
| **Replay** | Read raw transcript | Restore checkpoint state | Annotated step-by-step replay |
| **Citation** | No | No | Conclusions cite specific evidence steps |

```
Case #042 "Diagnose log-20260508.txt"
├── Step 1 | rag_search | "发动机故障 模式" | → P-042 matched | 0.3s
├── Step 2 | log_query  | keyword="ERR_THERMAL" | → 2 matches found | 1.2s
├── Step 3 | signal_query | engine_temp, -5s+10s | → 75→87°C chart | 0.5s
├── Step 4 | vision_analyze | chart | → "non-gradual rise" | 0.8s
└── Conclusion: "controller_overheat" (0.92)
    Evidence: [Step 1, Step 2, Step 3, Step 4]
```

### 3.10 Confidence + Citation

A middleware that evaluates the agent's output and attaches a confidence score
with citations to the evidence chain. Entirely optional.

| | Hermes | DeerFlow | Kairos |
|---|--------|----------|--------|
| **Confidence on output** | No | No (facts only) | ✅ Every agent output |
| **Citation** | No | No | ✅ Linked to Evidence Chain steps |
| **Low-confidence action** | No | No | ✅ Triggers clarification or human review |

```python
# Kairos output
{
    "conclusion": "Root cause: controller overheat",
    "confidence": 0.92,
    "evidence": [
        {"step": 1, "tool": "rag_search", "finding": "Pattern P-042 matched"},
        {"step": 2, "tool": "log_query", "finding": "ERR_THERMAL_001 at T+3.2s"},
        {"step": 3, "tool": "signal_query", "finding": "engine_temp 75→87°C"},
        {"step": 4, "tool": "vision_analyze", "finding": "Non-gradual rise confirmed"}
    ]
}
```

### 3.11 Skills + Curator

Inherited from Hermes. Skills are SKILL.md files (YAML frontmatter + Markdown)
that capture reusable procedures. Curator manages lifecycle: active → stale →
archived, with automatic backup before any destructive operation.

Kairos enhancement: skills are semantically indexed, so RAG can retrieve the
right skill without requiring an exact name match.

### 3.12 Model Providers

OpenAI-compatible abstraction. Any provider that speaks the OpenAI
chat completions format works. Multi-key credential pools with automatic
rotation on rate limits.

### 3.13 Session Search

SQLite + FTS5 full-text search across all historical sessions. Queryable
by keyword, date range, or tool use patterns.

### 3.14 Gateway (Multi-Platform)

Same agent, multiple messaging platforms: WeChat, Telegram, Discord, Slack,
Feishu, DingTalk, and more. The Gateway layer translates platform-specific
messages into the unified Agent interface.

### 3.15 RL Training

Atropos-compatible reinforcement learning pipeline. The agent records
trajectories (ShareGPT JSONL format). Training environments run multi-turn
rollouts with tool calling. Reward functions score the agent's output using
ToolContext — post-rollout access to the same terminal/filesystem the agent used.

### 3.16 Typed State

ThreadState pattern from DeerFlow: a typed dictionary extending the base
AgentState with structured fields. Reducers handle merge semantics
(e.g., `merge_artifacts` for deduplication).

### 3.17 Sandbox

Multi-provider sandbox abstraction: local process, Docker container,
SSH remote, cloud sandbox. Tools execute in the sandbox, not on the host.

### 3.18 Context Compression

Delegated to middleware. When token count nears the model's context window,
summarizes early messages while preserving recent context and maintaining
user/assistant role alternation.

---

## 4. Boundary: Framework vs Business

| Layer | Kairos (framework) provides | You (business) provide |
|-------|---------------------------|----------------------|
| Agent Loop | `while` loop + tool dispatch | — |
| Middleware | EvidenceTracker, ConfidenceScorer, ContextCompress, SkillLoader, ToolRateLimit | Custom middleware for your domain |
| Tools | `rag_search`, `knowledge_lookup` | `log_query`, `signal_query`, your domain tools |
| RAG | VectorStore, adapters, fusion engine | Knowledge base content (repair manuals, legal docs, product specs) |
| Knowledge | `KnowledgeSchema` base class, `KnowledgeStore` | `FaultDiagnosis`, `LegalCase`, your typed schemas |
| Evidence | Tracker, chain, replay | — |
| Confidence | Scorer middleware | — |
| Skills | Loader, Curator lifecycle | SKILL.md content |
| System Prompt | Template engine, auto-fill variables | Soul, role, response style, domain instructions |
| Chat CLI | `kairos chat` / `kairos run` | — |
| Sub-Agent | Factory, types, executor | Type definitions for your domain |
| Model | OpenAI-compatible abstraction | API keys |
| Gateway | Platform adapters | Platform credentials |
| Training | Environment registry, trajectory recording | Reward functions, datasets |
| Sandbox | Provider abstraction | — |
| State | ThreadState with reducers | Domain-specific state fields |

---

## 5. Comparison Matrix (All 18 Modules)

| # | Module | Hermes | DeerFlow | Kairos | Source |
|---|--------|:--:|:--:|:--:|--------|
| 1 | Agent Loop | ✅ while | ✅ LangGraph | ✅ while | Hermes |
| 2 | Middleware Pipeline | ❌ | ✅ 11-layer, 6 hooks | ✅ 6 hooks, 5 MVP layers | DeerFlow |
| 3 | Tool Registry | ✅ auto-discover | ✅ 4-source merge | ✅ auto-discover | Hermes |
| 4 | Chat CLI | ✅ chat + run | ❌ Web UI only | ✅ chat + run | Hermes |
| 5 | System Prompt | ✅ SOUL.md | ✅ modular template | ✅ modular template + hook | Kairos |
| 6 | Sub-Agent Factory | 🟡 delegate_task | ✅ typed factory | ✅ typed factory + evidence inheritance | DeerFlow + Kairos |
| 7 | RAG Engine | ❌ | ❌ | ✅ vector search + adapters + fusion | Kairos |
| 8 | Structured Knowledge | ❌ freeform only | 🟡 5 fixed categories | ✅ user-defined schemas | Kairos |
| 9 | Evidence Chain | ❌ flat transcript | ❌ state snapshots | ✅ Case→Step→Evidence | Kairos |
| 10 | Confidence + Citation | ❌ | ❌ | ✅ optional middleware | Kairos |
| 11 | Skills + Curator | ✅ lifecycle mgmt | ❌ | ✅ + semantic retrieval | Hermes + Kairos |
| 12 | Model Providers | ✅ 20+ | ✅ LiteLLM | ✅ OpenAI-compat | Hermes |
| 13 | Session Search | ✅ FTS5 | ❌ | ✅ FTS5 | Hermes |
| 14 | Gateway | ✅ 15+ platforms | ❌ | ✅ Phase 3 | Hermes |
| 15 | RL Training | ✅ Atropos | ❌ | ✅ Atropos-compat | Hermes |
| 16 | Typed State | ❌ | ✅ ThreadState | ✅ ThreadState | DeerFlow |
| 17 | Sandbox | ❌ host direct | ✅ multi-provider | ✅ multi-provider | DeerFlow |
| 18 | Context Compression | ✅ aggressive | ✅ LangChain | ✅ middleware | Both |

---

## 6. MVP Scope

### Phase 1 — Framework Foundation (now)

```
kairos/
├── core/
│   ├── loop.py              # Agent Loop
│   ├── middleware.py         # 6 hook types
│   └── state.py             # ThreadState
├── chat/
│   ├── cli.py               # kairos chat / kairos run
│   └── session.py           # Session management
├── prompt/
│   ├── template.py          # System prompt engine
│   └── defaults.py          # Default templates
├── tools/
│   ├── registry.py          # Self-registering tool registry
│   ├── base.py
│   ├── rag_search.py
│   └── knowledge_lookup.py
├── middleware/
│   ├── evidence.py          # Evidence Chain
│   ├── confidence.py        # Confidence + Citation
│   ├── compress.py          # Context compression
│   └── skill_loader.py      # Skill loading
├── agents/
│   ├── factory.py           # Sub-Agent factory
│   ├── types.py
│   └── executor.py
├── infra/
│   ├── rag/
│   │   ├── vector_store.py
│   │   └── adapters/
│   ├── knowledge/
│   │   ├── schema.py
│   │   └── store.py
│   └── evidence/
│       └── tracker.py
├── providers/
│   └── base.py
├── cli.py                   # Entry point: kairos
└── pyproject.toml
```

### Phase 2 — Advanced

```
+ skills/manager.py          # Skills + Curator
+ memory/store.py            # Persistent memory
+ session/search.py          # Session search (FTS5)
+ sandbox/                   # Sandbox abstraction
```

### Phase 3 — Production

```
+ gateway/platforms/         # Multi-platform messaging
+ training/                  # RL training pipeline
```
