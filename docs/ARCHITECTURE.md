# Kairos — Framework Architecture

> **Kairos** (καιρός): the decisive moment. An agent should act not whenever
> it can, but exactly when it should.

## 1. Design Philosophy

Kairos inherits from two frameworks and extends them in four directions:

| From | What | Why |
|------|------|-----|
| **Hermes** | Agent Loop, Tool Registry, Skills+Curator, Session Search, Gateway, RL Training | The most complete personal agent ecosystem |
| **DeerFlow** | Middleware Pipeline, Sub-Agent Factory, Sandbox, Typed State, Context Compression | The cleanest production-grade architecture |
| **Kairos (new)** | RAG Engine, Structured Knowledge, Evidence Chain, Confidence+Citation | Capabilities neither framework provides |

## 2. Module Map

```
┌──────────────────────────────────────────────────────────────┐
│                      Kairos Framework                         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │
│  │  Agent   │  │Middleware│  │  Tool    │  │   RAG        │ │
│  │  Loop    │  │ Pipeline │  │ Registry │  │   Engine     │ │
│  │ (Hermes) │  │(DeerFlow)│  │ (Hermes) │  │  (Kairos)   │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘ │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │
│  │ Knowledge│  │ Evidence │  │Confidence│  │   Model      │ │
│  │  Store   │  │  Chain   │  │ +Citation│  │  Providers   │ │
│  │ (Kairos) │  │ (Kairos) │  │ (Kairos) │  │  (Hermes)    │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘ │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │
│  │ Skills + │  │ Session  │  │  Gateway │  │  Sub-Agent   │ │
│  │ Curator  │  │  Search  │  │  多平台   │  │   Factory    │ │
│  │ (Hermes) │  │ (Hermes) │  │ (Hermes) │  │  (DeerFlow)  │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘ │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │ Sandbox  │  │  Typed   │  │   RL     │                  │
│  │(DeerFlow)│  │  State   │  │ Training │                  │
│  │          │  │(DeerFlow)│  │ (Hermes) │                  │
│  └──────────┘  └──────────┘  └──────────┘                  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. Module-by-Module Comparison

### 3.1 Agent Loop

| | Hermes | DeerFlow | Kairos |
|---|--------|----------|--------|
| **Implementation** | Hand-written `while` loop in `run_conversation()` | LangGraph `create_react_agent()` StateGraph | Hand-written `while` loop (from Hermes) |
| **Complexity** | ~100 lines of core logic | Graph abstraction with nodes + conditional edges | ~100 lines, synchronous first |
| **Extras** | Budget tracking, interrupt handling, compression triggers | Built-in checkpointing, graph visualization, conditional routing | Budget tracking from Hermes; compression delegated to middleware |
| **Why Kairos chose Hermes** | LangGraph adds dependency weight for marginal benefit. The while-loop is simple, debuggable, and sufficient. DeerFlow's checkpointing will be replicated via middleware. |

```
Kairos Agent Loop (simplified):

  messages = [system_prompt, user_message]
  while iterations < max_iterations:
      response = llm.chat(messages, tools)
      if response.tool_calls:
          for tc in response.tool_calls:
              result = execute_tool(tc.name, tc.args)
              messages.append(result)
              evidence.track_step(tc, result)     # ← Kairos evidence tracking
      else:
          confidence = scorer.evaluate(messages)   # ← Kairos confidence
          return response, confidence, evidence
```

### 3.2 Middleware Pipeline

| | Hermes | DeerFlow | Kairos |
|---|--------|----------|--------|
| **Has it?** | No | Yes, 11 layers | Yes, phased |
| **Design** | Cross-cutting concerns scattered in `run_agent.py` and `model_tools.py` | 6 hook types: `before_agent`, `after_agent`, `before_model`, `after_model`, `wrap_model_call`, `wrap_tool_call` | Same 6 hooks from DeerFlow |
| **MVP layers** | — | ThreadData → Uploads → Sandbox → DanglingToolCall → Summarization → Todo → Title → Memory → ViewImage → SubagentLimit → Clarification | 5 layers: EvidenceTracker → ContextCompress → ToolRateLimit → SkillLoader → Clarify |
| **Why Kairos chose DeerFlow** | A middleware pipeline is the single biggest architectural advantage DeerFlow has. It separates every cross-cutting concern into an independent, testable, composable layer. Kairos starts with 5 layers and grows as needed. |

```
Kairos Middleware Pipeline (MVP):

  EvidenceTracker      before_model  → inject case/step IDs into context
  ContextCompress      before_model  → summarize when near token limit
  ToolRateLimit        wrap_tool_call → prevent runaway tool calls
  SkillLoader          before_agent  → load relevant skills for the task
  Clarify              wrap_tool_call → intercept ask_user, return to caller
```

### 3.3 Tool Registry

| | Hermes | DeerFlow | Kairos |
|---|--------|----------|--------|
| **Registration** | `registry.register()` — self-registering, auto-discovered from `tools/*.py` | 4 sources merged: config tools + MCP + builtins + conditional | Same as Hermes: self-registering with `registry.register()` |
| **Schema** | OpenAI function-calling format | LangChain BaseTool | OpenAI function-calling format |
| **Discovery** | Automatic — any `tools/*.py` with `register()` at import time | Manual config or MCP dynamic discovery | Automatic (from Hermes) |
| **Why Kairos chose Hermes** | Simpler. No LangChain dependency. A tool is a Python function + a JSON schema. That's it. |

### 3.4 RAG Engine (Kairos Original)

| | Hermes | DeerFlow | Kairos |
|---|--------|----------|--------|
| **Knowledge retrieval** | No vector search. Skills loaded as Markdown text. | No vector search. Memory is JSON with mtime caching. | Full RAG pipeline with vector store |
| **External knowledge** | Not supported | Not supported | Pluggable knowledge base adapters |
| **Skill-as-knowledge** | Skills loaded by name, no semantic matching | N/A | Skills indexed for semantic retrieval |
| **Fusion** | No | No | Multi-source fusion: external KB + skills + session history |

**Why neither Hermes nor DeerFlow has this:**

Both treat knowledge as something you inject once (Hermes: MEMORY.md at session start; DeerFlow: `<memory>` block in system prompt). Neither supports "search knowledge → use result → search again with new context" — the iterative retrieval pattern that RAG requires.

**Kairos design:**

```
RAG Engine
├── retriever.py       # Vector search (ChromaDB / FAISS)
├── adapters/          # Knowledge base connectors
│   ├── markdown.py    # Local Markdown files
│   ├── pdf.py         # PDF documents
│   └── api.py         # External KB via REST
├── skills_index.py    # Semantic skill retrieval
└── fusion.py          # Merge results from multiple sources
```

### 3.5 Structured Knowledge Framework (Kairos Original)

| | Hermes | DeerFlow | Kairos |
|---|--------|----------|--------|
| **Knowledge format** | SKILL.md (freeform Markdown) | JSON with `category` and `confidence` fields | User-defined typed schemas |
| **Queryability** | grep by filename/title | Limited: facts are key-value | Full structured queries |
| **Schema** | Implicit (convention-based) | Fixed (5 fact categories) | Explicit, user-defined per domain |

**Why neither framework does this well:**

Hermes stores everything as unstructured Markdown. DeerFlow's facts have structure but only 5 fixed categories (`preference`, `knowledge`, `context`, `behavior`, `goal`). Neither supports domain-specific schemas like:

```
FaultRecord:
  signal_name: str
  signal_value: float
  log_pattern: str
  root_cause: str
  solution: str
  confidence: float
```

**Kairos design:**

```python
# Framework provides the interface
class KnowledgeSchema(BaseModel):
    """Base class for domain knowledge schemas."""
    id: str
    created_at: datetime
    updated_at: datetime

# Business defines the schema
class FaultDiagnosis(KnowledgeSchema):
    signal_name: str
    log_pattern: str
    root_cause: str
    solution: str

# Framework provides the store
store = KnowledgeStore(schema=FaultDiagnosis)
store.insert(fault)
results = store.query({"root_cause": "controller_overheat"})
```

### 3.6 Evidence Chain (Kairos Original)

| | Hermes | DeerFlow | Kairos |
|---|--------|----------|--------|
| **Step tracking** | Session transcript (flat message list) | LangGraph checkpoint (state snapshots) | Structured Case → Step → Evidence chain |
| **Replay** | Read session log | Restore from checkpoint | Replay with step-by-step annotation |
| **Citation** | No | No | Every conclusion cites its evidence steps |

**Why neither framework does this:**

Hermes saves full conversation transcripts but they're flat — you can't easily extract "step 3 used tool X which produced result Y which led to conclusion Z". DeerFlow's checkpoints are state snapshots for recovery, not for audit trails.

**Kairos design:**

```
Evidence Chain
├── tracker.py     # Case objects with ordered Step records
│   Case
│   ├── step_1: tool=rag_search, args={...}, result={...}, duration=0.3s
│   ├── step_2: tool=log_query, args={...}, result={...}, duration=1.2s
│   ├── step_3: tool=signal_query, args={...}, result={...}, duration=0.5s
│   └── step_4: tool=vision_analyze, args={...}, result={...}, duration=0.8s
│       → conclusion: "controller_overheat", confidence: 0.92
│       → evidence: [step_2, step_3, step_4]
└── replay.py      # Reconstruct reasoning from evidence chain
```

### 3.7 Confidence + Citation (Kairos Original)

| | Hermes | DeerFlow | Kairos |
|---|--------|----------|--------|
| **Output confidence** | No | No (facts have confidence, outputs don't) | Every output carries confidence score |
| **Evidence citation** | No | No | Every conclusion cites its evidence |
| **Low-confidence handling** | No | No | Triggers clarification or human review |

```
Hermes output:
  "The root cause is a controller overheat."

Kairos output:
  {
    "conclusion": "Root cause: controller overheat",
    "confidence": 0.92,
    "evidence": [
      {"step": 2, "tool": "log_query", "finding": "ERR_THERMAL_001 at T+3.2s"},
      {"step": 3, "tool": "signal_query", "finding": "temp_signal = 87.3°C"},
      {"step": 4, "tool": "knowledge_match", "finding": "Pattern P-042: overheat"}
    ]
  }
```

### 3.8 Skills + Curator

| | Hermes | DeerFlow | Kairos |
|---|--------|----------|--------|
| **Skill format** | SKILL.md (YAML frontmatter + Markdown) | N/A | Same as Hermes |
| **Lifecycle** | Curator: active → stale → archived | N/A | Same as Hermes |
| **Evolution** | Agent patches skills when they break | N/A | Same as Hermes |
| **Enhancement** | — | — | Skills indexed for semantic retrieval via RAG |

**Why Kairos keeps Hermes' approach:**

Hermes' skill system is the best in class. The Curator lifecycle management (pin → stale → archive → backup) is thoughtful. Kairos adds one enhancement: skills are semantically indexed so RAG can retrieve the right skill even when not called by exact name.

### 3.9 Sub-Agent Factory

| | Hermes | DeerFlow | Kairos |
|---|--------|----------|--------|
| **Delegation** | `delegate_task` tool | `task` tool | `task` tool (from DeerFlow) |
| **Types** | leaf / orchestrator | general-purpose / bash | Configurable types from DeerFlow |
| **Tool restriction** | Via toolsets | Blacklist (gp) or whitelist (bash) | Configurable per type |
| **Concurrency** | Configurable (default 3) | Hard capped [2,4] | Configurable with hard cap |
| **Timeout** | None | 900s configurable | Configurable |
| **Context** | Isolated | Isolated | Isolated |
| **Sandbox** | Independent terminal | Shared with parent (lazy_init) | Configurable |
| **Evidence tracking** | No | No | Each sub-agent gets its own evidence chain |

**Why Kairos chose DeerFlow:**

DeerFlow's typed sub-agent factory (general-purpose with full tools, bash with minimal tools) is a cleaner pattern than Hermes' toolsets approach. Kairos extends it with evidence chain inheritance — parent can see which sub-agent discovered what.

### 3.10 RL Training

| | Hermes | DeerFlow | Kairos |
|---|--------|----------|--------|
| **Framework** | Atropos integration | None | Atropos-compatible (from Hermes) |
| **Environments** | Terminal, SWE, Web Research | None | Extensible environment registry |
| **Trajectory format** | ShareGPT JSONL | None | Same as Hermes |
| **ToolContext** | Post-rollout tool access for reward computation | None | Same as Hermes |
| **Model serving** | VLLM / SGLang managed | None | Same as Hermes |
| **Why Kairos keeps this** | An agent framework that can train the models it runs on closes the loop. This is Hermes' most unique capability and Kairos preserves it fully. |

---

## 4. What Kairos Creates That Neither Has

| Capability | Why Hermes doesn't have it | Why DeerFlow doesn't have it |
|-----------|--------------------------|------------------------------|
| **RAG Engine** | Knowledge is flat Markdown, no vector search | Memory is assistant context, not a searchable knowledge base |
| **Structured Knowledge** | Skills are freeform, no typed schemas | Facts are limited to 5 fixed categories |
| **Evidence Chain** | Sessions are transcripts, not structured traces | Checkpoints are for recovery, not audit |
| **Confidence + Citation** | Output is plain text | Confidence exists only on facts, not outputs |

These four capabilities are framework-level — they apply to any domain, any task, any Agent built on Kairos. They're not vehicle-diagnosis-specific. They exist because Hermes and DeerFlow were designed before the current generation of LLMs made structured reasoning tractable.

---

## 5. MVP Scope

### Phase 1 (framework foundation)

```
kairos/
├── core/
│   ├── loop.py           # Agent Loop (Hermes)
│   ├── middleware.py      # Middleware hooks (DeerFlow)
│   └── state.py           # Typed state (DeerFlow)
├── tools/
│   ├── registry.py        # Tool registry (Hermes)
│   └── base.py            # Tool base class
├── rag/                   # RAG Engine (Kairos)
│   ├── retriever.py
│   └── store.py
├── knowledge/             # Structured Knowledge (Kairos)
│   ├── schema.py
│   └── store.py
├── evidence/              # Evidence Chain (Kairos)
│   ├── tracker.py
│   └── chain.py
├── confidence/            # Confidence + Citation (Kairos)
│   └── scorer.py
├── providers/
│   └── base.py            # Model abstraction
├── cli.py                 # Entry point
└── pyproject.toml
```

### Phase 2 (advanced capabilities)

```
+ skills/manager.py        # Skills + Curator (Hermes)
+ memory/store.py          # Persistent memory
+ session/search.py        # Session search (Hermes)
+ agents/factory.py        # Sub-agent factory (DeerFlow)
+ middleware/builtins/      # 5+ middleware layers
+ sandbox/                 # Sandbox abstraction (DeerFlow)
```

### Phase 3 (production)

```
+ gateway/platforms/       # Multi-platform (Hermes)
+ training/env.py          # RL training (Hermes)
+ context/compressor.py    # Context compression (Hermes + DeerFlow)
```
