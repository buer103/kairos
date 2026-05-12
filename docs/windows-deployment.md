# Kairos on Windows — Deployment Guide

Three ways to deploy Kairos on Windows, from simplest to production-grade.

## Option 1: Direct Python Install (simplest)

For development, scripting, and single-machine usage.

### Prerequisites

```powershell
# Install Python 3.10+ from https://python.org
# Check: python --version

# Recommended: create a virtual environment
python -m venv kairos-env
kairos-env\Scripts\activate
```

### Install

```powershell
pip install kairos-agent

# Optional extras:
# pip install kairos-agent[gateway]      # Gateway HTTP server
# pip install kairos-agent[rag]          # RAG / vector search
# pip install kairos-agent[dev]          # Development tools + tests
# pip install kairos-agent[all]          # Everything
```

### Configure

```powershell
# Set your API key in environment
$env:DEEPSEEK_API_KEY = "sk-your-key"

# Or: create config file (paths auto-adapt to Windows)
python -m kairos config init
# Config saved to: C:\Users\<you>\.config\kairos\config.yaml
```

### Run

```powershell
# Interactive chat
kairos chat

# One-shot query
kairos "Explain quantum computing"

# As a library
python -c "from kairos import Agent; from kairos.providers.base import ModelConfig;
agent = Agent(model=ModelConfig(api_key='$env:DEEPSEEK_API_KEY'));
print(agent.run('Hello')['content'])"
```

### Windows-specific notes

| Feature | Status | Note |
|---------|--------|------|
| Agent Loop | ✅ | Full support |
| CLI chat mode | ✅ | Works, no tab completion (readline unavailable) |
| Tab completion | ⚠️ | Install `pip install pyreadline3` to enable |
| Gateway server | ✅ | Full support (aiohttp works on Windows) |
| Local sandbox | ✅ | Executes in local shell |
| Docker sandbox | ✅ | Requires Docker Desktop |
| SSH sandbox | ✅ | Windows 10+ has built-in `ssh` client |
| Signal handlers | ✅ | Already guarded with try/except |
| Config file | ✅ | Auto-uses `%USERPROFILE%\.config\kairos\` |


## Option 2: Docker (recommended for production)

Consistent environment, no Python version conflicts, easy to scale.

### Prerequisites

- [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
- Git (optional, for cloning)

### Build & Run

```powershell
# Clone
git clone https://github.com/buer103/kairos.git
cd kairos

# Build the image
docker build -t kairos-agent .

# Run with API key
docker run -it --rm `
  -e DEEPSEEK_API_KEY=sk-your-key `
  -p 8080:8080 `
  kairos-agent

# Or: use docker-compose
docker-compose up -d
```

### Environment variables

```powershell
docker run -it --rm `
  -e DEEPSEEK_API_KEY=sk-your-key `
  -e KAIROS_DEBUG=true `
  -e KAIROS_GATEWAY_PORT=8080 `
  -v ${PWD}/config.yaml:/home/kairos/.config/kairos/config.yaml `
  -v ${PWD}/data:/home/kairos/.kairos `
  kairos-agent
```

### docker-compose.yml (Windows paths)

```yaml
version: "3.8"
services:
  kairos:
    image: kairos-agent
    build: .
    ports:
      - "8080:8080"
    environment:
      - DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}
      - KAIROS_DEBUG=true
    volumes:
      # Windows absolute paths with forward slashes
      - C:/Users/YourName/kairos-config:/home/kairos/.config/kairos
      - C:/Users/YourName/kairos-data:/home/kairos/.kairos
    restart: unless-stopped
```

### Health check

```powershell
curl http://localhost:8080/health
# {"status": "healthy", "agent": "Kairos", "version": "0.16.0"}
```


## Option 3: Development Setup

For contributing to Kairos or customizing the source.

```powershell
# Clone and install in editable mode
git clone https://github.com/buer103/kairos.git
cd kairos
pip install -e ".[dev]"

# Run tests
pytest tests/ -q

# Verify
python -c "import kairos; print(kairos.__version__)"
# 0.16.0
```

### IDE setup (VS Code)

```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/kairos-env/Scripts/python.exe",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/", "-q"]
}
```

## Limitations on Windows

These features work but with caveats:

| Feature | Limitation | Workaround |
|---------|-----------|------------|
| Tab completion | `readline` not available | `pip install pyreadline3` |
| Cron scheduler | Unix signals unavailable | Use Docker or Windows Task Scheduler |
| Docker sandbox | Needs Docker Desktop | Install Docker Desktop (free) |
| File paths in config | Backslash vs forward slash | Use forward slashes or raw strings |
| Large file watching | `watchdog` may miss events on network drives | Keep files on local disk |

## Quick Decision Matrix

| Use case | Recommended option |
|----------|-------------------|
| "I just want to try it" | Option 1 — `pip install` |
| "I want to build a chatbot service" | Option 2 — Docker |
| "I want to extend Kairos" | Option 3 — dev setup |
| "I need to integrate with my FastAPI app" | Option 1 — `pip install kairos-agent` |
| "I need cron jobs" | Option 2 — Docker (Unix inside container) |
