# Changelog

All notable changes to Kairos will be documented in this file.

## [v0.15.0-dev] — 2026-05-10

### Added

- **Trace ID full-chain observability** — span tree with trace_id/span_id/parent_span_id, JSONL persistence, context-var based implicit propagation across sub-agents, `get_trace()` / `list_traces()` API (21 tests)
- **Tiered Memory (3 tiers)** — DeerFlow-compatible Profile/Timeline/Facts with confidence≥0.7 filtering, per-agent isolation, 2000 token injection budget, TTL expiry (34 tests)
- **Provider Registry** — 7 built-in profiles (DeepSeek, OpenRouter, Groq, Qwen, OpenAI, Anthropic, Gemini), auto-detection of native SDK vs OpenAI-compat, env var fallback (35 tests)
- **Documentation** — rewritten README with full stats, complete ARCHITECTURE.md with 62-module tree and 30-feature comparison matrix, CHANGELOG.md

### Changed

- Memory system upgraded from flat key-value to 3-tier with confidence filter
- Provider plugin now reads from centralized ProviderRegistry (was 4 hardcoded entries)
- Tests: 605 → 695 (+90)

## [v0.15.0-dev] — 2026-05-09

### Added

- **Config Schema** — Pydantic validation for 10 config sections (14 tests)
- **Health + Readiness Endpoints** — `/health` liveness, `/ready` readiness (503 on issues), `/health/detailed` component-status (7 tests)
- **Gateway Webhook Signature Verification** — 6 platforms: Discord Ed25519, Feishu/WhatsApp HMAC-SHA256, Line base64-HMAC, Matrix keyed-HMAC (18 tests)
- **Docker Deployment** — Multi-stage Dockerfile + docker-compose with HEALTHCHECK
- **Graceful Shutdown** — Gateway + Webhook session drain and queue drain
- **Config + Memory Tests** — 34 tests covering config loading, memory CRUD, legacy adapters

### Changed

- Tests: 566 → 605 (+39)
- Commits: 36 → 41

## [v0.14.0] — 2026-05-08

### Added

- **Gateway Deep Upgrade** — 11 platform adapters production-grade (Telegram 356L, WeChat 376L XML+签名, Slack 611L Block Kit, Discord/Signal/Matrix/IRC/Feishu/WhatsApp/Line)
- **CLI/TUI Enhancement** — 10 skins (default/hacker/retro/minimal/ocean/sunset/forest/midnight/neon/mono), tab completion, setup wizard
- **Browser Tools** — web_scrape, web_screenshot, web_search_advanced (4 engines), web_form_submit with proxy auto-detection
- **MCP Client** — JSON-RPC 2.0 over stdio, subprocess lifecycle management
- **Vision Tools** — vision_analyze, vision_compare, vision_screenshot_analyze

### Expanded Test Coverage

- Providers: 56 tests
- Agents: 61 tests
- Training: 43 tests
- Plugins: 42 tests
- CLI: 31 tests
- Total: 299 → 566 (+89%)

## [v0.13.0] — 2026-05-07

### Added

- **Sub-agent Orchestrator** — DelegationManager with depth-limited delegation tree, recursive worker spawning
- **Context Compression v3** — TrajectoryCompressor + ImportanceScorer, LLM-powered summarization, greedy selection within token budget

## [v0.12.0] — 2026-05-06

### Added

- **Memory FTS5 Full-Text Search** — SQLite FTS5 + BM25 ranking
- **Tool Ecosystem v2** — 23 tools: browser, MCP, vision

## [v0.11.0] — 2026-05-05

### Added

- **Security Layer** — 6 modules: FileSafety, URLSafety, PathSecurity, ContentRedactor, Guardrails
- **Observability Layer** — ErrorClassifier (sliding window), UsageTracker (token/cost), AgentInsights (health reports)
- **Agent Loop v3** — ProviderFactory + CredentialPool rotation + ModelHealth fallback + Trajectory

## [v0.10.0] — 2026-05-04

- Agent Loop v2: budget control, interrupt/resume, checkpoint, error classification
- Provider/Tools/Gateway upgrades
- 299 tests

## [v0.9.0] — 2026-05-03

- Skill Marketplace: install from GitHub/HF/URL/local
- True SSE streaming (3-layer: provider → loop → gateway)
- CLI enhancement (4 skins, 10+ slash commands)

## [v0.8.0] — 2026-05-02

- Config system (YAML/JSON, env fallback)
- CLI integration
- README + pyproject.toml
- 137 tests

## [v0.7.0] — 2026-05-01

- 11-platform Gateway
- Layered context compression
- 137 tests

## [v0.6.0] — 2026-04-30

- Cron scheduler (SQLite-backed)
- Rich TUI (spinner, panels, slash commands)
- Sandbox wiring
- Sub-agent delegation
- 121 tests

## [v0.5.0] — 2026-04-29

- Structured logging (JSON formatter, rotation)
- Plugin architecture (6 types)
- Tool expansion
- 100 tests

## [v0.4.0] — 2026-04-28

- P0/P1 production-grade completion
- 100 tests

## [v0.3.0] — 2026-04-27

- Full framework delivery: 18 modules
- Middleware pipeline aligned to DeerFlow 11 layers (5→14 layers)
- 76 tests

## [v0.2.0] — 2026-04-26

- Phase 1: Prompt engine, RAG infra, Knowledge store, Evidence DB
- Middleware refactor, Sub-Agent factory

## [v0.1.0] — 2026-04-25

- Initial framework: Agent Loop, Middleware Pipeline, Tool Registry
- Model Providers, CLI, built-in tools
- Architecture design: 18 modules, 3-layer classification
