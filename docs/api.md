# Kairos API Reference

> v0.16.0 — 20-layer middleware pipeline agent framework

## Quick Start

```python
from kairos import Agent
from kairos.providers.base import ModelConfig

agent = Agent.build_default(model=ModelConfig(api_key="sk-..."))
response = agent.run("Explain quantum computing in one paragraph")
print(response["content"])
```

## Core

### `Agent` — one-shot query agent

```python
from kairos import Agent
from kairos.providers.base import ModelConfig

# Full pipeline (20 middleware layers)
agent = Agent.build_default(
    model=ModelConfig(api_key="...", base_url=None, model="gpt-4o"),
    agent_name="MyAgent",
    role_description="You are a helpful assistant.",
    max_iterations=20,
    max_tokens=120000,
    skills_dir="./skills",
    is_plan_mode=True,
    memory_store=backend,
    supports_vision=True,
    enable_security=True,
    credential_pool=pool,
    enable_insights=True,
)

# Minimal agent (no middleware)
agent = Agent(model=ModelConfig(api_key="..."))

result = agent.run("What is 2+2?")
# result: {"content": "4", "tool_calls": [], "usage": {...}, "iterations": 1}
```

### `StatefulAgent` — multi-turn chat with session persistence

```python
from kairos import StatefulAgent
from kairos.providers.base import ModelConfig

agent = StatefulAgent(model=ModelConfig(api_key="..."))
response1 = agent.chat("Hello, my name is Alice")
response2 = agent.chat("What is my name?")  # remembers context

# Streaming
for event in agent.chat_stream("Tell a story"):
    if event["type"] == "token":
        print(event["content"], end="")

# Session persistence
agent.save_session("my-session")
agent.load_session("my-session")
```

### `ThreadPaths` — workspace directory management

```python
from kairos.core.paths import ThreadPaths

paths = ThreadPaths.ensure()
print(paths.workspace)  # ~/.kairos/threads/<thread_id>/
print(paths.data)       # ~/.kairos/threads/<thread_id>/data/
```

---

## Middleware Pipeline (20 layers)

The `Agent.build_default()` assembles these layers in dependency order:

| # | Middleware | Hook | Purpose |
|---|-----------|------|---------|
| 0 | ThreadDataMiddleware | before_agent | Workspace dirs + thread_id injection |
| 1 | UploadsMiddleware | before_agent | Injects uploaded files into context |
| 2 | DanglingToolCallMiddleware | wrap_model_call | Fixes broken/malformed tool calls |
| 3 | SkillLoader | before_agent | Loads skill documents into system prompt |
| 4 | ContextCompressor | before_model | Token budget management + LLM summarization |
| 5 | TodoMiddleware | before_model | Persists todo list across compression |
| 6 | MemoryMiddleware | before_model | Injects Profile/Timeline/Facts into context |
| 7 | ViewImageMiddleware | before_model | Attaches images for vision-capable models |
| 8 | EvidenceTracker | wrap_tool_call | Records evidence chain per step |
| 9 | ToolArgRepairMiddleware | wrap_model_call | Repairs broken JSON in tool arguments |
| 10 | SandboxAuditMiddleware | wrap_tool_call | Blocks dangerous terminal commands |
| 11 | SecurityMiddleware | wrap_tool_call | Path/URL guardrails (optional, `enable_security`) |
| 12 | ConfidenceScorer | after_model | 6-factor output quality scoring |
| 13 | LLMRetryMiddleware | wrap_model_call | 3-state circuit breaker + credential rotation |
| 14 | LoopDetectionMiddleware | after_model | Breaks infinite tool-call loops |
| 15 | SubagentLimitMiddleware | after_model | Caps concurrent sub-agent calls |
| 16 | TokenUsageMiddleware | after_model | Per-message token/cost attribution |
| 17 | TitleMiddleware | after_model | Auto-generates session title |
| 18 | ClarificationMiddleware | wrap_tool_call | Intercepts ask_user (MUST be last) |
| 19 | MemoryMiddleware (write) | after_agent | Submits to memory write queue |

**Custom middleware:**

```python
from kairos.core.middleware import Middleware

class ProfilerMiddleware(Middleware):
    def after_model(self, state, runtime):
        latency = runtime.get("last_latency_ms", 0)
        print(f"[Profiler] Turn {runtime['turn']}: {latency}ms")
        return None

agent = Agent(model=config, middlewares=[ThreadDataMiddleware(), ProfilerMiddleware()])
```

---

## Tools

### Built-in tools

| Tool | Description |
|------|-------------|
| `read_file` | Read file with line numbers |
| `write_file` | Create/overwrite file |
| `search_files` | Regex content search or glob file search |
| `terminal` | Execute shell commands |
| `web_search` | Web search via DuckDuckGo |
| `web_fetch` | Fetch and parse web pages |
| `rag_search` | Vector similarity search in RAG store |
| `knowledge_lookup` | Query structured knowledge stores |
| `vision_analyze` | Analyze images via vision models |
| `session_search` | FTS5 full-text search across past sessions |
| `skill_view` | Load skill content |
| `skills_list` | List available skills |
| `skill_manage` | Create/patch/delete skills |
| `delegate_task` | Spawn sub-agents for parallel work |
| `memory` | Save/retrieve durable facts |
| `clarify` | Ask user for clarification |
| `todo` | Manage session task list |
| `browser_navigate` | Navigate browser to URL |
| `browser_screenshot` | Capture browser screenshot |

### Custom tools

```python
from kairos import register_tool

@register_tool
def get_weather(city: str, units: str = "celsius") -> dict:
    """Get current weather for a city.

    Args:
        city: City name (e.g., 'Tokyo', 'London')
        units: Temperature units ('celsius' or 'fahrenheit')
    """
    # Your business logic here
    return {"city": city, "temp": 22, "condition": "sunny"}

# Tool auto-registered — Agent can now call get_weather
agent = Agent.build_default(model=config)
response = agent.run("What's the weather in Tokyo?")
```

---

## Memory (3-tier)

```python
from kairos.memory import MemoryStore, MemoryMiddleware
from kairos.memory import ProfileMemory, TimelineMemory, FactMemory

store = MemoryStore()

# Tier 1: Profile — overwrite semantics (language, role, preferences)
store.profile.save("language", "Chinese")
store.profile.save("role", "backend engineer")

# Tier 2: Timeline — append-only, timestamped events
store.timeline.append("tool_call", {"tool": "read_file", "path": "main.py"})
store.timeline.append("user_query", {"query": "fix the bug"})

# Tier 3: Facts — durability-filtered (confidence ≥ 0.7), auto-TTL
store.facts.save("prefers_concise", "User prefers short responses", confidence=0.9)
store.facts.save("project_dir", "~/workspace/kairos", confidence=0.8)

# Search
results = store.search("user prefers")  # FTS5 full-text
agent = Agent.build_default(model=config, memory_store=store)
```

---

## Skills

```python
from kairos.skills import SkillManager, SkillStatus

manager = SkillManager(skills_dir="~/.kairos/skills")

# List installed skills
for skill in manager.list():
    print(f"{skill.name} [{skill.status.value}]")

# Install from marketplace
manager.install("github.com/user/my-skill")
manager.install("huggingface.co/org/skill")
manager.install("https://example.com/skill.tar.gz")
manager.install("~/my-skills/custom-skill")

# Curator lifecycle
manager.curator.status()       # active/stale/archived counts
manager.curator.clean(days=30) # remove stale backups + archived
manager.curator.reindex()      # full rescan
```

---

## Gateway (12 platforms)

```python
from kairos.gateway import GatewayServer, UnifiedMessage, UnifiedResponse

# Start gateway with all platform adapters
server = GatewayServer(
    agent=agent,
    telegram_token="...",
    wechat_token="...",
    slack_token="...",
    discord_token="...",
)
await server.start(host="0.0.0.0", port=8080)

# Endpoints:
#  GET  /health           — health check
#  GET  /metrics          — Prometheus metrics
#  POST /chat             — chat query
#  GET  /chat/stream      — SSE streaming
#  POST /webhook/telegram — Telegram webhook
#  POST /webhook/wechat   — WeChat webhook
```

### Platform adapter reference

| Adapter | Webhook | Signatures |
|---------|:--:|------|
| TelegramAdapter | ✅ getUpdates polling | — |
| WeChatAdapter | ✅ XML parse + reply | SHA1 signature check |
| SlackAdapter | ✅ Events API | HMAC-SHA256 |
| DiscordAdapter | ✅ Gateway Intents | Ed25519 |
| WhatsAppAdapter | ✅ Webhook | HMAC-SHA256 |
| SignalAdapter | ✅ Webhook | — |
| FeishuAdapter | ✅ Event Subscription | HMAC-SHA256 |
| LineAdapter | ✅ Webhook | HMAC-SHA256 |
| MatrixAdapter | ✅ Sync loop | — |
| IRCAdapter | ✅ TCP connection | — |
| CLIAdapter | N/A | — |

---

## Providers (17 profiles)

```python
from kairos.providers.base import ModelConfig
from kairos.providers.registry import ProviderRegistry, get_provider

# Direct config
config = ModelConfig(
    api_key="sk-...",
    base_url="https://api.openai.com/v1",  # or https://api.deepseek.com/v1
    model="gpt-4o",                         # or deepseek-chat, claude-sonnet-4
)

# Provider registry
profile = get_provider("anthropic")  # auto-detects native SDK
config = ModelConfig(api_key="sk-ant-...", model="claude-sonnet-4-20250514")
provider = profile.create(config)

# Supported profiles:
# openai, deepseek, anthropic, gemini, openrouter, groq, qwen,
# mistral, together, perplexity, cohere, xai, replicate, azure,
# cloudflare, fireworks, huggingface
```

### Credential pool (multi-key rotation)

```python
from kairos.providers.credential import CredentialPool, Credential, RetryConfig

pool = CredentialPool([
    Credential(api_key="sk-key1", provider="openai"),
    Credential(api_key="sk-key2", provider="openai"),
    Credential(api_key="sk-key3", provider="openai"),
])
# Auto-rotates on 429 rate-limit, marks keys as exhausted
agent = Agent.build_default(model=config, credential_pool=pool)
```

### Provider failover

```python
from kairos.providers.failover import FailoverConfig
# When primary provider fails (429/auth/timeout), auto-degrades to fallbacks
agent = Agent.build_default(
    model=primary_config,
    fallback_models=[fallback_config1, fallback_config2],
)
```

---

## Sub-Agents

```python
from kairos.agents.delegate import (
    DelegateTask, DelegationManager, DelegateConfig, register_delegate_tool
)

manager = DelegationManager(
    model=model,
    config=DelegateConfig(
        max_concurrent=3,        # hard cap on parallel sub-agents
        default_timeout=180,     # seconds per sub-agent
        policy="whitelist",      # 'whitelist', 'blacklist', or a preset name
        allowed_tools=["read_file", "terminal"],
    ),
)

# Run in parallel
results = manager.run_parallel([
    DelegateTask(goal="Analyze file A", context="...", tools=["read_file"]),
    DelegateTask(goal="Analyze file B", context="...", tools=["read_file"]),
    DelegateTask(goal="Analyze file C", context="...", tools=["read_file"]),
])

# Register in agent tools
register_delegate_tool(manager)
```

---

## Security

### 3-Level Permission System

```python
from kairos.security.permission import PermissionManager, PermissionLevel, PermissionAction

pm = PermissionManager(auto_approve=False)

# BLOCK: always deny (e.g. cronjob)
# ASK: prompt user (e.g. terminal, write_file)  
# TRUST: always allow (e.g. read_file, search_files)

# Set per-tool policies
from kairos.security.permission import ToolPolicy
pm.set_policy(ToolPolicy("write_file", level=PermissionLevel.ASK))
pm.set_policy(ToolPolicy("read_file", level=PermissionLevel.TRUST))

# Interactive CLI prompt (y/n/session-grant)
# In CLI mode, SecurityMiddleware auto-activates with Rich-based prompt
# /yolo → toggles auto_approve (bypass all checks)
# /perm trust <tool> → runtime policy change
# /perm block <tool>
# /perm show → list all policies
```

### Sandbox

```python
from kairos.sandbox import LocalSandbox, DockerSandbox, SSHSandbox

sandbox = DockerSandbox(image="python:3.12-slim", memory_limit="512m")
result = sandbox.run("python -c 'print(2+2)'")
# result: {"stdout": "4\n", "stderr": "", "exit_code": 0, "duration_ms": 120}
```

### Guardrails

```python
from kairos.security.guardrails import InputGuard, OutputGuard
from kairos.security.file_safety import FileSafetyChecker

guard = InputGuard(max_length=50000)
ok, reason = guard.validate_input(user_message)

checker = FileSafetyChecker()
ok, reason = checker.is_safe("/etc/passwd", kind="path")
```

---

## Hook System (24 lifecycle hooks)

```python
from kairos.hooks import HookPoint, get_hook_registry

registry = get_hook_registry()

# Register a callback (priority: lower = earlier execution)
@registry.on(HookPoint.BEFORE_TOOL, priority=50)
def audit_tool_calls(tool_name, args, **ctx):
    print(f"Tool: {tool_name}({args})")

# Emit manually
registry.emit(HookPoint.BEFORE_TOOL, tool_name="read_file", args={"path": "x.py"})

# All 24 hook points:
#   Agent: AGENT_START, AGENT_END, AGENT_INTERRUPT
#   Model: BEFORE_MODEL, AFTER_MODEL, MODEL_ERROR
#   Tool:  BEFORE_TOOL, AFTER_TOOL, TOOL_ERROR, TOOL_RETRY
#   Message: BEFORE_MESSAGE, AFTER_MESSAGE
#   Session: SESSION_SAVE, SESSION_LOAD
#   Middleware: MIDDLEWARE_CHAIN_START, MIDDLEWARE_CHAIN_END
#   Compress: BEFORE_COMPRESSION, AFTER_COMPRESSION
#   Memory: MEMORY_SAVE, MEMORY_LOAD
#   Gateway: GATEWAY_MESSAGE_RECEIVED, GATEWAY_RESPONSE_SENT
#   Skills: SKILL_LOADED, SKILL_CREATED

# Query
print(registry.listeners(HookPoint.BEFORE_TOOL))  # 1
print(registry.list_hooks())  # {"tool:before": 1, ...}
```

---

## Observability

```python
from kairos.observability import UsageTracker, ErrorClassifier

# Token usage + cost tracking
tracker = UsageTracker()
tracker.track_call(input_tokens=500, output_tokens=200, model="gpt-4o")
print(tracker.session_cost())  # $0.0011

# Error classification (5 categories)
classifier = ErrorClassifier()
classifier.record_error("rate_limit", provider="openai")
classifier.record_error("auth", provider="anthropic")
print(classifier.health_report())
# {"rate_limit": 1, "auth": 1, "healthy": True}

# Agent health insights
agent = Agent.build_default(model=config, enable_insights=True)
result = agent.run("query")
print(agent.health_status())
# {"total_calls": 1, "avg_latency_ms": 850, "error_rate": 0.0}
```

---

## Session Search

```python
from kairos.session import SessionSearch

search = SessionSearch()
results = search.query("quantum computing")  # FTS5 full-text
for r in results:
    print(f"[{r['thread_id']}] {r['title']} — {r['timestamp']}")
```

---

## RL Training

```python
from kairos.training import (
    TrajectoryRecorder, TrainingEnv, reward_confidence, reward_success_rate
)

# Record trajectories for training
recorder = TrajectoryRecorder(output_dir="./trajectories")
agent = Agent.build_default(model=config, trajectory_dir="./trajectories")

# Custom reward functions
env = TrainingEnv(
    rewards=[reward_confidence, reward_success_rate, reward_evidence_quality]
)
```

---

## CLI

```bash
# Interactive chat (Rich CLI)
kairos --api-key sk-... --model gpt-4o

# Textual TUI (multi-panel)
kairos tui                    # default
kairos tui --skin hacker      # hacker theme

# Custom provider (vLLM, Ollama, etc.)
kairos --api-key not-needed --base-url http://localhost:8000/v1 --model llama-3

# Web UI
kairos web                    # starts at http://127.0.0.1:8080
kairos web --port 3000

# Health check
kairos doctor

# Config management
kairos config init            # interactive setup wizard
kairos config migrate         # upgrade config with new defaults

# Session management
kairos sessions list
kairos sessions resume <id>

# Skill management
kairos skill list
kairos skill install github.com/user/repo
kairos skill view <name>
```

### Slash commands (interactive mode)

| Command | Action |
|---------|--------|
| `/yolo` | Skip dangerous command confirmation |
| `/edit` | Multi-line input for code/long text |
| `/retry` | Resend last message |
| `/undo` | Pop last exchange |
| `/background <query>` | Run query in background |
| `/goal <text>` | Set persistent goal |
| `/goal status/pause/resume/clear` | Manage goal |
| `/reasoning` | Toggle reasoning/thinking display |
| `/session rename/delete` | Session management |
| `/help` | Show all commands |

---

## Textual TUI

```bash
# Launch Textual TUI — multi-panel terminal chat interface
kairos tui                           # default skin
kairos tui --skin hacker             # hacker (green-on-black)
kairos tui --skin retro              # retro (cyan/magenta)
kairos tui --verbose                 # show tool arguments
kairos tui --resume my-session       # resume saved session
kairos tui --model gpt-4o            # specify model
```

The Textual TUI provides a multi-panel layout with sidebar, streaming chat,
tool call cards, and skin hot-switching — all in pure Python with no
Node.js dependency.

**Layout:**
```
┌──────────┬──────────────────────────────────────┐
│ Sidebar  │  Header (thinking… model)            │
│ Sessions │  Transcript (streaming chat)         │
│ Tools    │  ─  ⏳ terminal (running)            │
│ Model    │  ─  ✅ search_files (12ms)          │
│          │  Status (tokens/cost)                │
│          │  Input bar                           │
└──────────┴──────────────────────────────────────┘
```

**Key bindings:**

| Key | Action |
|-----|--------|
| `Ctrl+C` | Interrupt running agent |
| `Ctrl+R` | Retry last message |
| `Ctrl+L` | Clear transcript |
| `Ctrl+S` | Save current session |
| `Ctrl+Q` | Quit |
| `Escape` | Clear input |

**Slash commands:**

| Command | Action |
|---------|--------|
| `/help` | Show all commands |
| `/clear` | Clear transcript |
| `/skin <name>` | Switch skin (default/hacker/retro/minimal) |
| `/model <name>` | Switch model |
| `/tools` | List available tools |
| `/sessions` | List saved sessions |
| `/save <name>` | Save session |
| `/load <name>` | Load session |
| `/verbose` | Toggle verbose tool output |
| `/retry` | Retry last message |
| `/undo` | Undo last exchange |
| `/yolo` | Toggle YOLO mode |
| `/reasoning` | Toggle reasoning display |
| `/edit` | Multi-line input mode |

**Programmatic:**

```python
from kairos.tui import KairosTUI
from kairos.core.stateful_agent import StatefulAgent
from kairos.providers.base import ModelConfig

agent = StatefulAgent(model=ModelConfig(api_key="...", model="deepseek-chat"))
app = KairosTUI(agent=agent, skin="default", verbose=False)
app.run()
```

---

## Config System

```yaml
# ~/.config/kairos/config.yaml
model:
  api_key: sk-...
  base_url: https://api.openai.com/v1
  model: gpt-4o

agent:
  name: Kairos
  max_iterations: 20
  max_tokens: 120000

gateway:
  host: 0.0.0.0
  port: 8080

skills:
  dir: ~/.kairos/skills
  external_dirs: []

memory:
  backend: sqlite
  path: ~/.kairos/memory/memory.db

observability:
  log_level: INFO
  metrics_port: 9090

security:
  level: ask
  allowed_paths: []
```

```python
from kairos.config import Config, get_config

config = get_config()
print(config.model.api_key)
print(config.agent.max_iterations)
```

---

## Web UI

```bash
# Start the web server (embedded SPA)
kairos web

# Custom host/port
kairos web --host 0.0.0.0 --port 3000
```

The Web UI is a single-page application with:
- **Multi-session** sidebar — create/switch/delete sessions
- **Live streaming** — real-time token-by-token output via SSE
- **Rich tool cards** — per-tool icons, status badges, timing, expandable args
- **Markdown rendering** — code blocks, links, lists, blockquotes
- **Copy-to-clipboard** — one-click on any agent message
- **Mobile responsive** — hamburger menu, overlay backdrop

### Grace Call Retry

```python
# Automatic retry on tool failure with simplified arguments
# In Agent._execute_loop(), tool execution wraps with _grace_call():
#   - Truncates long string args
#   - Drops optional parameters
#   - Retries once with fallback defaults
#   - Emits TOOL_RETRY hook on success
```

### Sub-Agent CancelEvent

```python
from kairos.agents.subagent import CancelEvent

cancel = CancelEvent()
cancel.set()  # signal cancellation
print(cancel.is_set())  # True
```
