# Kairos — Framework Architecture

> **Kairos** (καιρός): the decisive moment. An agent should act not whenever
> it can, but exactly when it should.

## 1. Design Philosophy

Kairos inherits from two proven frameworks and extends them in four directions:

| From | What | Why |
|------|------|-----|
| **Hermes** | Agent Loop, Tool Registry, Skills+Curator, Session Search, Gateway, RL Training, Chat CLI | The most complete personal agent ecosystem |
| **DeerFlow** | Middleware Pipeline, Sub-Agent Factory, Sandbox, Typed State, Context Compression | The cleanest production-grade architecture |
| **Kairos (new)** | RAG Engine, Structured Knowledge, Evidence Chain, Confidence+Citation | Capabilities neither framework provides |

---

## 2. Three Roles

Kairos is designed to serve three distinct roles simultaneously:

```
┌─────────────────────────────────────────────┐
│              Kairos Framework               │
│                                             │
│  Role 1: Standalone CLI                     │
│  $ kairos chat                              │
│  $ kairos run "诊断故障"                     │
│                                             │
│  Role 2: Python Library                     │
│  from kairos import Agent                   │
│  agent = Agent(tools=..., knowledge=...)    │
│  agent.run("诊断故障")                       │
│                                             │
│  Role 3: Business Platform                  │
│  YourAgent(Kairos) → 诊断系统 / RAG系统 / ... │
│  Kairos provides the loop, tools, middleware │
│  You provide the domain tools and knowledge  │
└─────────────────────────────────────────────┘
```

---

## 3. Module Map

```
┌──────────────────────────────────────────────────────────────────┐
│                        Kairos Framework                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │  Agent   │  │Middleware│  │  Tool    │  │   Chat   │        │
│  │  Loop    │  │ Pipeline │  │ Registry │  │   CLI    │        │
│  │ (Hermes) │  │(DeerFlow)│  │ (Hermes) │  │ (Hermes) │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │   RAG    │  │Knowledge │  │ Evidence │  │Confidence│        │
│  │  Engine  │  │  Store   │  │  Chain   │  │ +Citation│        │
│  │ (Kairos) │  │ (Kairos) │  │ (Kairos) │  │ (Kairos) │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ Skills + │  │  Model   │  │ Sub-Agent│  │ Session  │        │
│  │ Curator  │  │Providers │  │ Factory  │  │  Search  │        │
│  │ (Hermes) │  │ (Hermes) │  │(DeerFlow)│  │ (Hermes) │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │  Typed   │  │ Sandbox  │  │ Gateway  │  │   RL     │        │
│  │  State   │  │(DeerFlow)│  │ (Hermes) │  │ Training │        │
│  │(DeerFlow)│  │          │  │          │  │ (Hermes) │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Three-Layer Classification

Every capability in Kairos falls into exactly one of these layers:

| Layer | Rule | Examples |
|-------|------|----------|
| **Tool** | Agent actively decides when to call it | `rag_search`, `knowledge_lookup`, user-defined domain tools |
| **Middleware** | Framework auto-executes at specific lifecycle hooks | EvidenceTracker, ConfidenceScorer, ContextCompress |
| **Infrastructure** | Underlying storage/engine used by both tools and middleware | VectorStore, KnowledgeStore, EvidenceTrackerDB |

```
Agent Loop (core, unchanged)
    │
    ├── Tools: Agent calls proactively
    │   └── e.g. "I need to search the knowledge base → call rag_search()"
    │
    ├── Middleware: Framework intercepts passively
    │   └── e.g. "A tool was called → EvidenceTracker records it"
    │
    └── Infrastructure: Shared by both
        └── e.g. EvidenceTracker writes to EvidenceTrackerDB
             rag_search() reads from VectorStore
```

---

## 4. Module-by-Module Comparison

### 4.1 Agent Loop

| | Hermes | DeerFlow | Kairos |
|---|--------|----------|--------|
| **Implementation** | Hand-written `while` loop | LangGraph `create_react_agent()` | Hand-written `while` loop (from Hermes) |
| **Complexity** | ~100 lines of core logic | Graph abstraction with nodes + edges | ~100 lines |
| **Extras** | Budget tracking, interrupt handling, compression triggers | Built-in checkpointing, graph visualization | Budget tracking from Hermes; compression via middleware |
| **Why Kairos chose Hermes** | LangGraph adds dependency weight for marginal benefit. The while-loop is simple, debuggable, and sufficient. |

The Agent Loop itself stays **pure** — it only handles message passing, LLM calls, and tool dispatch. All enhancements (evidence tracking, confidence scoring, RAG) happen in middleware or tools, not in the loop.

### 4.2 Middleware Pipeline

| | Hermes | DeerFlow | Kairos |
|---|--------|----------|--------|
| **Has it?** | No | Yes, 11 layers, 6 hook types | Yes, 6 hook types from DeerFlow |
| **Design** | Cross-cutting concerns scattered | `before_agent`, `after_agent`, `before_model`, `after_model`, `wrap_model_call`, `wrap_tool_call` | Same 6 hooks |
| **MVP layers** | — | 11 layers | 5 layers |

```
Kairos Middleware Pipeline (MVP):

  SkillLoader          before_agent   → load relevant skills by semantic match
  EvidenceTracker      before_model   → inject case/step IDs into context
  ContextCompress      before_model   → summarize when near token limit
  ToolRateLimit        wrap_tool_call → prevent runaway tool calls
  ConfidenceScorer     after_agent    → evaluate confidence, attach evidence
```

**Why Kairos chose DeerFlow:** A middleware pipeline is the single biggest architectural advantage DeerFlow has. It separates every cross-cutting concern into an independent, testable, composable layer. All middleware is optional — disable what you don't need.

### 4.3 Tool Registry

| | Hermes | DeerFlow | Kairos |
|---|--------|----------|--------|
| **Registration** | `registry.register()` — auto-discovered from `tools/*.py` | 4 sources merged: config + MCP + builtins + conditional | Same as Hermes |
| **Schema** | OpenAI function-calling format | LangChain BaseTool | OpenAI function-calling format |
| **Why Kairos chose Hermes** | Simpler. A tool is a Python function + a JSON schema. No LangChain dependency. |

### 4.4 Chat CLI

| | Hermes | DeerFlow | Kairos |
|---|--------|----------|--------|
| **Interactive chat** | `hermes chat` | Web UI only | `kairos chat` |
| **Single query** | `hermes chat -q "..."` | API only | `kairos run "..."` |
| **Tool loading** | Via config | Via config | `kairos chat --tools my_tools.py` |
| **Knowledge loading** | — | — | `kairos chat --knowledge fault_schema.py` |
| **Session resume** | `hermes --resume` | Via thread_id | `kairos chat --resume <session_id>` |

```
$ kairos chat
🤖 Kairos> 诊断 log-20260508.txt
         [rag_search] → 找到故障模式 P-042
         [log_query]  → 发现 ERR_THERMAL_001
         [signal_query] → 温度 87.3°C
         
         根因: 控制器过热
         置信度: 0.92
         证据: [Step1] [Step2] [Step3]

$ kairos run "诊断 log-20260508.txt"
{ "conclusion": "控制器过热", "confidence": 0.92, "evidence": [...] }
```

### 4.5 Sub-Agent Factory

Kairos takes DeerFlow's typed factory model and adds evidence chain inheritance.

| | Hermes delegate_task | DeerFlow task | Kairos task |
|---|------|------|------|
| **Delegation** | `delegate_task` tool | `task` tool | `task` tool |
| **Types** | leaf / orchestrator | general-purpose / bash | User-definable types |
| **Tool restriction** | Via toolsets | Blacklist or whitelist | Blacklist or whitelist, per type |
| **Concurrency** | Configurable (default 3) | Hard capped [2,4] | Configurable with hard cap |
| **Timeout** | None | 900s configurable | Configurable |
| **Context** | Isolated | Isolated | Isolated |
| **Sandbox** | Independent terminal | Shared (lazy_init) | Configurable |
| **Evidence** | No | No | ✅ Sub-chain → parent chain |

```
Lead Agent (interacting with user)
    │
    │  task(description="Analyze segment 01", subagent_type="diagnose")
    ├── Sub-Agent A
    │   └── EvidenceChain(sub_A): Step1 → Step2 → finding: anomaly_A
    │
    │  task(description="Analyze segment 02", subagent_type="diagnose")  
    ├── Sub-Agent B
    │   └── EvidenceChain(sub_B): Step1 → finding: no anomaly
    │
    │  task(description="Analyze segment 03", subagent_type="diagnose")
    └── Sub-Agent C
        └── EvidenceChain(sub_C): Step1 → Step2 → finding: anomaly_B

Lead Agent synthesizes:
    EvidenceChain(parent)
    ├── sub_A: anomaly_A (overheat signal)
    ├── sub_B: no anomaly
    └── sub_C: anomaly_B (voltage drop)
    → Conclusion: combined anomaly_A + anomaly_B → root cause
```

### 4.6 RAG Engine (Kairos Original)

| | Hermes | DeerFlow | Kairos |
|---|--------|----------|--------|
| **Knowledge retrieval** | No vector search. Skills loaded as Markdown text. | No vector search. Memory is JSON. | Full RAG pipeline with vector store |
| **External knowledge** | Not supported | Not supported | Pluggable knowledge base adapters |
| **Fusion** | No | No | Multi-source: external KB + skills + session history |

RAG is implemented as two parts:
- **Infrastructure**: `VectorStore`, adapters (Markdown, PDF, API), fusion strategies
- **Tool**: `rag_search(query, top_k)` — the Agent actively calls this when it needs knowledge

### 4.7 Structured Knowledge Framework (Kairos Original)

| | Hermes | DeerFlow | Kairos |
|---|--------|----------|--------|
| **Format** | Freeform Markdown | 5 fixed fact categories | User-defined typed schemas |
| **Query** | grep | Limited key-value | Full structured queries |

Hermes stores everything as unstructured Markdown. DeerFlow's facts have structure but only 5 fixed categories (`preference`, `knowledge`, `context`, `behavior`, `goal`). Kairos provides a `KnowledgeSchema` base class — users define their own typed schemas per domain.

```python
# Framework
class KnowledgeSchema(BaseModel):
    id: str
    created_at: datetime

# Business
class FaultDiagnosis(KnowledgeSchema):
    signal_name: str
    log_pattern: str
    root_cause: str
    solution: str

store = KnowledgeStore(schema=FaultDiagnosis)
store.query({"root_cause": "controller_overheat"})
```

### 4.8 Evidence Chain (Kairos Original)

| | Hermes | DeerFlow | Kairos |
|---|--------|----------|--------|
| **Step tracking** | Flat transcript | State snapshots (recovery) | Structured Case → Step → Evidence |
| **Replay** | Read session log | Restore checkpoint | Annotated replay |
| **Citation** | No | No | Conclusions cite evidence steps |

Evidence Chain is a middleware — the Agent Loop doesn't know it exists:

```python
class EvidenceTracker:
    def wrap_tool_call(self, request, handler):
        result = handler(request)
        self.db.record_step(
            case_id=self.current_case,
            tool=request.name,
            args=request.args,
            result=result,
            duration=elapsed
        )
        return result
```

### 4.9 Confidence + Citation (Kairos Original)

| | Hermes | DeerFlow | Kairos |
|---|--------|----------|--------|
| **Output confidence** | No | Facts only, not outputs | Every output |
| **Evidence citation** | No | No | Linked to evidence chain |
| **Optional** | — | — | Disable via middleware config |

Confidence is a middleware — optional, composable:

```python
class ConfidenceScorer:
    def after_agent(self, state, runtime):
        score = self.evaluate(state.messages, state.evidence)
        return {
            "confidence": score,
            "evidence": self.tracker.get_chain()
        }
```

This means:
- Diagnostic agents get confidence scores
- Chat agents skip it entirely
- Same framework, different configs

### 4.10 Skills + Curator

Kairos keeps Hermes' skill system exactly, with one enhancement:

| | Hermes | Kairos |
|---|--------|--------|
| **Format** | SKILL.md | Same |
| **Lifecycle** | Curator: active → stale → archived | Same |
| **Evolution** | Agent patches broken skills | Same |
| **Semantic retrieval** | By exact name only | ✅ RAG-indexed for semantic matching |

### 4.11 Remaining Modules

| Module | Source | Kairos treatment |
|--------|--------|-----------------|
| **Model Providers** | Hermes | OpenAI-compatible abstraction, unchanged |
| **Session Search** | Hermes | SQLite FTS5, unchanged |
| **Gateway (multi-platform)** | Hermes | Phase 3, unchanged |
| **RL Training** | Hermes | Atropos-compatible, unchanged |
| **Sandbox** | DeerFlow | Multi-provider abstraction, Phase 2 |
| **Typed State** | DeerFlow | ThreadState pattern, Phase 1 |
| **Context Compression** | Both | Delegated to middleware |

---

## 5. Boundary: Framework vs Business

| Layer | Kairos provides | Business provides |
|-------|----------------|-------------------|
| **Agent Loop** | while + tool dispatch | — |
| **Middleware** | EvidenceTracker, ConfidenceScorer, ContextCompress, SkillLoader, ToolRateLimit | Custom middleware |
| **Tools** | `rag_search`, `knowledge_lookup` | `log_query`, `signal_query`, domain tools |
| **RAG** | VectorStore, adapters, fusion | Knowledge base content |
| **Knowledge** | Schema base class, Store | FaultDiagnosis schema, LegalCase schema |
| **Skills** | Loader, Curator | Skill content |

---

## 6. What Kairos Creates That Neither Has

| Capability | Why Hermes doesn't have it | Why DeerFlow doesn't have it |
|-----------|--------------------------|------------------------------|
| **RAG Engine** | Knowledge is flat Markdown, no vector search | Memory is assistant context, not a searchable KB |
| **Structured Knowledge** | Skills are freeform | Facts limited to 5 fixed categories |
| **Evidence Chain** | Transcripts are flat | Checkpoints are for recovery, not audit |
| **Confidence + Citation** | Output is plain text | Confidence exists on facts, not outputs |

These are **framework-level**, universal across all domains. They exist because Hermes and DeerFlow were designed before structured reasoning with citation became practical.

---

## 7. MVP Scope

### Phase 1 — Framework Foundation

```
kairos/
├── core/
│   ├── loop.py              # Agent Loop (Hermes)
│   ├── middleware.py         # 6 hook types (DeerFlow)
│   └── state.py             # Typed state (DeerFlow)
├── chat/                    # CLI interface (Hermes)
│   ├── cli.py              # kairos chat / kairos run
│   └── session.py          # Session management
├── tools/
│   ├── registry.py          # Tool registry (Hermes)
│   ├── base.py
│   ├── rag_search.py        # RAG retrieval tool
│   └── knowledge_lookup.py  # Knowledge query tool
├── middleware/
│   ├── evidence.py          # Evidence Chain (Kairos)
│   ├── confidence.py        # Confidence + Citation (Kairos)
│   ├── compress.py          # Context compression
│   └── skill_loader.py      # Skill loading
├── agents/                  # Sub-Agent factory (DeerFlow)
│   ├── factory.py
│   ├── types.py
│   └── executor.py
├── infra/                   # Infrastructure (Kairos)
│   ├── rag/
│   │   ├── vector_store.py
│   │   └── adapters/
│   ├── knowledge/
│   │   ├── schema.py
│   │   └── store.py
│   └── evidence/
│       └── tracker.py
├── providers/
│   └── base.py              # Model abstraction
├── cli.py                   # Entry point: kairos command
└── pyproject.toml
```

### Phase 2 — Advanced

```
+ skills/                    # Skills + Curator (Hermes)
+ memory/                    # Persistent memory
+ session/                   # Session search (Hermes)
+ sandbox/                   # Sandbox abstraction (DeerFlow)
```

### Phase 3 — Production

```
+ gateway/                   # Multi-platform (Hermes)
+ training/                  # RL training (Hermes)
```
