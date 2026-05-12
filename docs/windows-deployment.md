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


## Option 3: 源码安装（推荐开发用）

从 GitHub clone 源码，可编辑安装，方便修改和调试。

### 第一步：clone + 安装

```powershell
# 1. Clone 源码
git clone https://github.com/buer103/kairos.git
cd kairos

# 2. 创建虚拟环境（推荐）
python -m venv venv
venv\Scripts\activate

# 3. 可编辑安装（改了源码立即生效）
pip install -e ".[dev]"

# 4. 验证
python -c "import kairos; print(kairos.__version__)"
# 输出: 0.16.0
```

### 第二步：配置 API Key

```powershell
# 方式 A: 环境变量（最简单）
$env:DEEPSEEK_API_KEY = "sk-your-key"

# 方式 B: 写入配置文件
$env:DEEPSEEK_API_KEY = "sk-your-key"
kairos config init
# 配置文件: C:\Users\<你>\.config\kairos\config.yaml
```

### 第三步：启动

安装成功后，`kairos` 命令已注册到 PATH，有 **4 种启动方式**：

```powershell
# ── 方式 1: 交互式对话（最常用） ──
kairos chat

# ── 方式 2: 一次性查询 ──
kairos "深圳今天天气怎么样"
kairos run "解释一下 Kubernetes 调度器"

# ── 方式 3: 启动 Gateway HTTP 服务 ──
# 先装依赖: pip install kairos-agent[gateway]
python -m kairos.gateway
# 然后在浏览器打开 http://localhost:8080/health

# ── 方式 4: 作为 Python 库嵌入你的代码 ──
python -c "
from kairos import Agent
from kairos.providers.base import ModelConfig
agent = Agent(model=ModelConfig(api_key='sk-...'))
print(agent.run('Hello')['content'])
"
```

### 第四步：跑通测试（确认环境正常）

```powershell
pytest tests/ -q
# 1,300+ tests passed
```

### VS Code 配置

```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/Scripts/python.exe",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/", "-q"]
}
```

### 源码安装的额外能力

| 能力 | 命令 |
|------|------|
| 会话管理 | `kairos --list-sessions`，`kairos --resume <name>` |
| Cron 定时任务 | `kairos cron list`，`kairos cron add <name> <cron>` |
| 技能管理 | `kairos skill list`，`kairos skill install <url>` |
| Curator 清理 | `kairos curator status`，`kairos curator clean` |
| 查看版本 | `kairos --version` |

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
| "I want to modify the source" | Option 3 — 源码安装 |
| "I want to build a chatbot service" | Option 2 — Docker |
| "I need to integrate with my FastAPI app" | Option 1 — `pip install kairos-agent` |
| "I need cron jobs" | Option 2 — Docker (Unix inside container) |
