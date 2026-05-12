# Deploying Kairos in Production

This guide covers deploying Kairos Agent as a production HTTP API service.

## Prerequisites

- Docker & Docker Compose (recommended)
- API key for at least one supported provider (DeepSeek, OpenAI, Anthropic, etc.)
- 1+ GB RAM, 2+ CPU cores recommended

## Quick Deploy (Docker Compose)

```bash
# Clone the repo
git clone https://github.com/buer103/kairos.git
cd kairos

# Copy and edit the config
cp config.yaml.example config.yaml
# Edit config.yaml — set your preferred provider

# Set your API key
export DEEPSEEK_API_KEY=sk-your-key-here

# Start the service
docker compose up -d

# Check health
curl http://localhost:8080/health
# {"status": "ok", "uptime_seconds": 5.2, "sessions": 0}

# Send a test request
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, what can you do?"}'
```

## Deploy Options

### Option 1: Docker Compose (Recommended)

```yaml
# docker-compose.prod.yml
version: "3.8"
services:
  kairos:
    build: .
    restart: unless-stopped
    ports:
      - "8080:8080"
    environment:
      - DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}
      - KAIROS_LOG_LEVEL=INFO
      - KAIROS_PROVIDER=deepseek
    volumes:
      - kairos_data:/home/kairos/.kairos
      - ./config.yaml:/home/kairos/.config/kairos/config.yaml:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 5s
      retries: 3

volumes:
  kairos_data:
```

### Option 2: Direct Python

```bash
pip install kairos-agent
# or: pip install -e .

export DEEPSEEK_API_KEY=sk-...
python -m kairos.gateway --host 0.0.0.0 --port 8080 --provider deepseek
```

### Option 3: Systemd Service

```ini
# /etc/systemd/system/kairos.service
[Unit]
Description=Kairos Agent Gateway
After=network.target

[Service]
Type=simple
User=kairos
WorkingDirectory=/opt/kairos
Environment=DEEPSEEK_API_KEY=sk-...
Environment=KAIROS_LOG_LEVEL=INFO
ExecStart=/opt/kairos/venv/bin/python -m kairos.gateway --port 8080
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now kairos
sudo systemctl status kairos
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chat` | Chat completion (sync) |
| `GET` | `/chat/stream?message=...` | Streaming chat (SSE) |
| `GET` | `/health` | Liveness probe |
| `GET` | `/ready` | Readiness probe (checks model availability) |
| `GET` | `/health/detailed` | Component-level health status |
| `GET` | `/stats` | Request/error/session statistics |
| `POST` | `/sessions/clear` | Clear all active sessions |

### Chat Request

```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain quantum computing in one paragraph",
    "session_id": "user-123"
  }'
```

Response:
```json
{
  "content": "Quantum computing uses qubits...",
  "confidence": 0.92,
  "evidence": [...],
  "interrupted": false
}
```

### Streaming Chat

```bash
curl -N "http://localhost:8080/chat/stream?message=Tell%20me%20a%20story"
```

SSE events:
```
data: {"type":"token","content":"Once"}
data: {"type":"token","content":" upon"}
data: {"type":"token","content":" a"}
...
data: {"type":"done","content":"Once upon a time...","confidence":0.88}
```

## Health & Monitoring

### Health endpoints

```bash
# Liveness — is the process running?
curl http://localhost:8080/health
# → {"status":"ok","uptime_seconds":3600.0,"sessions":5}

# Readiness — can it serve requests? (503 if model unavailable)
curl http://localhost:8080/ready
# → {"status":"ready","checks":{"agent_loaded":true,"model_configured":true}}

# Detailed — component-level diagnostics
curl http://localhost:8080/health/detailed
# → {"status":"healthy","components":{"agent":{...},"gateway":{...}}}
```

### Docker healthcheck

The Dockerfile includes a `HEALTHCHECK` directive that calls `/health` every 30 seconds. Docker Compose will auto-restart if unhealthy.

### Logs

```bash
# Docker
docker compose logs -f kairos

# Systemd
journalctl -u kairos -f
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEEPSEEK_API_KEY` | — | DeepSeek API key |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `OPENROUTER_API_KEY` | — | OpenRouter API key |
| `GROQ_API_KEY` | — | Groq API key |
| `DASHSCOPE_API_KEY` | — | Qwen/DashScope API key |
| `GEMINI_API_KEY` | — | Gemini API key |
| `KAIROS_PROVIDER` | `deepseek` | Default provider |
| `KAIROS_LOG_LEVEL` | `INFO` | Log level |
| `KAIROS_SESSION_TTL` | `3600` | Session timeout (seconds) |
| `KAIROS_REQUEST_TIMEOUT` | `300` | Request timeout (seconds) |

## Scaling

For production workloads:

1. **Multiple instances** behind a load balancer (NGINX, HAProxy)
2. **Shared state**: Sessions are in-memory by default. For multi-instance, use Redis-backed sessions (future)
3. **Rate limiting**: Gateway includes sliding-window rate limiter
4. **Provider fallback**: Configure `fallback_providers` in config.yaml for resilience

## Troubleshooting

### Gateway won't start

```bash
# Check the provider is configured
python -c "from kairos.providers.registry import list_providers; print([p['name'] for p in list_providers()])"

# Check the API key env var is set
echo $DEEPSEEK_API_KEY

# Run with debug logging
KAIROS_LOG_LEVEL=DEBUG python -m kairos.gateway
```

### Health check fails

```bash
# Verify the gateway is listening
curl -v http://localhost:8080/health

# Check if the agent model is configured
curl http://localhost:8080/ready
# → {"status":"not_ready",...} means model key is missing or invalid
```

### Rate limiting

If you see 429 responses, check:
- `gateway.rate_limit.requests_per_minute` in config.yaml
- Provider rate limits (DeepSeek: 500/min, OpenAI: varies by tier)
- Credential pool stats for key exhaustion

## Deploying the Web UI

The Web UI is a lightweight single-page application embedded in the Python package.
No npm, no build step, no extra dependencies.

```bash
# Start the Web UI server
kairos web --host 0.0.0.0 --port 8080

# Or via Docker
docker run -d -p 8080:8080 \\
  -e DEEPSEEK_API_KEY=$DEEPSEEK_API_KEY \\
  kairos:latest web --host 0.0.0.0 --port 8080
```

Access at `http://localhost:8080` — multi-session chat with live streaming,
rich tool cards, markdown rendering, and mobile-responsive design.

### Health check endpoints

```
GET /api/health          → {"status":"ok","uptime":...,"sessions":3}
GET /api/sessions        → {"sessions":[...]}
GET /api/sessions/active → {"active":["default","session-a1b2"],"count":2}
```
