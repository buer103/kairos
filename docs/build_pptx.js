const PptxGenJS = require("pptxgenjs");
const pptx = new PptxGenJS();

// ═══════════════════════════════════════════════════════
// THEME — Terminal-inspired dark
// ═══════════════════════════════════════════════════════
const BG    = "0D1117";
const BG2   = "161B22";
const BG3   = "21262D";
const CYAN  = "58A6FF";
const GREEN = "3FB950";
const YELLOW= "D29922";
const RED   = "F85149";
const DIM   = "8B949E";
const WHITE = "C9D1D9";
const BORDER= "30363D";

pptx.defineLayout({ name:"CUSTOM", width:13.33, height:7.5 });
pptx.layout = "CUSTOM";

// Helper: add a terminal-style code block
function addTerminal(slide, x, y, w, h, lines, opts={}) {
  const fontSize = opts.fontSize || 11;
  const lineH = opts.lineH || 0.32;
  slide.addShape(pptx.ShapeType.rect, {
    x, y, w, h, fill: { color: BG }, line: { color: BORDER, width: 0.5 },
    rectRadius: 0.05
  });
  // Title bar
  slide.addShape(pptx.ShapeType.rect, {
    x, y, w: w, h: 0.3, fill: { color: BG3 },
    rectRadius: 0.05
  });
  slide.addText("● ● ●", { x: x+0.15, y: y+0.02, w: 0.5, h: 0.26, fontSize: 8, color: DIM });
  slide.addText(opts.title || "kairos — terminal", {
    x: x+0.6, y: y+0.02, w: w-0.8, h: 0.26, fontSize: 8, color: DIM, align: "center"
  });
  // Content
  const textY = y + 0.4;
  lines.forEach((line, i) => {
    if (Array.isArray(line)) {
      // Array of {text, color} segments
      slide.addText(line, {
        x: x+0.15, y: textY + i*lineH, w: w-0.3, h: lineH,
        fontSize: fontSize, fontFace: "Consolas", valign: "top"
      });
    } else {
      slide.addText(line, {
        x: x+0.15, y: textY + i*lineH, w: w-0.3, h: lineH,
        fontSize: fontSize, fontFace: "Consolas", color: DIM, valign: "top"
      });
    }
  });
}

// Helper: slide background
function setBg(slide, color) {
  slide.background = { fill: color || BG };
}

// ═══════════════════════════════════════════════════════
// SLIDE 1 — Title
// ═══════════════════════════════════════════════════════
{
  const s = pptx.addSlide();
  setBg(s, BG);
  // Kairos logo
  s.addText("καιρός", {
    x: 0, y: 1.8, w: "100%", h: 1.2,
    fontSize: 64, fontFace: "Georgia", color: CYAN, align: "center", bold: true
  });
  s.addText("Kairos Agent Framework  v0.16.0", {
    x: 0, y: 3.0, w: "100%", h: 0.6,
    fontSize: 22, fontFace: "Calibri", color: WHITE, align: "center"
  });
  s.addText("The right tool, at the right moment.", {
    x: 0, y: 3.6, w: "100%", h: 0.4,
    fontSize: 14, fontFace: "Calibri", color: DIM, align: "center", italic: true
  });
  // Stats row
  const stats = [
    { num: "1,276", label: "Tests" },
    { num: "20", label: "Middleware Layers" },
    { num: "17", label: "LLM Providers" },
    { num: "11", label: "Platform Adapters" },
    { num: "10", label: "TUI Skins" },
  ];
  const startX = 1.5;
  stats.forEach((st, i) => {
    const sx = startX + i * 2.2;
    s.addText(st.num, {
      x: sx, y: 4.5, w: 1.8, h: 0.7,
      fontSize: 36, fontFace: "Georgia", color: GREEN, align: "center", bold: true
    });
    s.addText(st.label, {
      x: sx, y: 5.2, w: 1.8, h: 0.35,
      fontSize: 12, fontFace: "Calibri", color: DIM, align: "center"
    });
  });
  s.addText("github.com/buer103/kairos", {
    x: 0, y: 6.6, w: "100%", h: 0.35,
    fontSize: 11, color: DIM, align: "center"
  });
}

// ═══════════════════════════════════════════════════════
// SLIDE 2 — Interactive Chat Session
// ═══════════════════════════════════════════════════════
{
  const s = pptx.addSlide();
  setBg(s, BG);
  s.addText("Interactive Chat Session", {
    x: 0.5, y: 0.25, w: 12, h: 0.55,
    fontSize: 26, fontFace: "Georgia", color: CYAN, bold: true
  });
  s.addText("Live streaming output with Rich TUI — token-by-token rendering", {
    x: 0.5, y: 0.8, w: 12, h: 0.3, fontSize: 12, color: DIM
  });

  // Terminal mockup
  const lines = [
    [{text:"ℹ️  Kairos 0.16.0 · Model: deepseek-chat · Streaming: ON", options:{color:DIM,fontSize:10}}],
    [{text:""}],
    [{text:"You", options:{color:GREEN,bold:true,fontSize:10}}],
    [{text:"  分析一下项目架构", options:{color:WHITE,fontSize:10}}],
    [{text:""}],
    [{text:"🤖 Kairos", options:{color:CYAN,bold:true,fontSize:10}}],
    [{text:"  好的，让我查看项目结构...", options:{color:WHITE,fontSize:10}}],
    [{text:""}],
    [{text:"  🔧 search_files (120ms)", options:{color:YELLOW,fontSize:10}}],
    [{text:"    pattern: *.py  target: files", options:{color:DIM,fontSize:9}}],
    [{text:""}],
    [{text:"  这是一个三层架构的 AI Agent 框架:", options:{color:WHITE,fontSize:10}}],
    [{text:"  1. Agent Loop — ReAct 模式循环", options:{color:WHITE,fontSize:10}}],
    [{text:"  2. Middleware Pipeline — 20 层中间件", options:{color:WHITE,fontSize:10}}],
    [{text:"  3. Infrastructure — RAG/Knowledge/Evidence", options:{color:WHITE,fontSize:10}}],
    [{text:""}],
    [{text:"  📊 Tokens: 1,847 · in: 1,234 · out: 613 · ≈$0.0048", options:{color:DIM,fontSize:9}}],
  ];
  addTerminal(s, 0.3, 1.3, 7.5, 5.8, lines, {title:"kairos — bash"});

  // Right side: features
  const features = [
    { icon: "🔄", title: "Live Streaming", desc: "Token-by-token output\nusing Rich Live display" },
    { icon: "🔧", title: "Tool Calls", desc: "Visual tool execution\nwith timing & results" },
    { icon: "📊", title: "Token Tracking", desc: "Usage + cost shown\nafter every response" },
    { icon: "📌", title: "Status Bar", desc: "Model · Session\nTokens · Cost" },
  ];
  features.forEach((f, i) => {
    const fy = 1.5 + i * 1.35;
    s.addShape(pptx.ShapeType.rect, {
      x: 8.3, y: fy, w: 4.5, h: 1.2, fill: { color: BG2 },
      line: { color: BORDER, width: 0.5 }, rectRadius: 0.05
    });
    s.addText(f.icon, { x: 8.5, y: fy+0.1, w: 0.5, h: 0.5, fontSize: 20, align:"center" });
    s.addText(f.title, {
      x: 9.1, y: fy+0.1, w: 3.5, h: 0.35,
      fontSize: 13, fontFace: "Calibri", color: CYAN, bold: true
    });
    s.addText(f.desc, {
      x: 9.1, y: fy+0.5, w: 3.5, h: 0.6,
      fontSize: 11, fontFace: "Calibri", color: DIM
    });
  });
}

// ═══════════════════════════════════════════════════════
// SLIDE 3 — Slash Commands
// ═══════════════════════════════════════════════════════
{
  const s = pptx.addSlide();
  setBg(s, BG);
  s.addText("Slash Commands", {
    x: 0.5, y: 0.25, w: 12, h: 0.55,
    fontSize: 26, fontFace: "Georgia", color: CYAN, bold: true
  });
  s.addText("14 built-in commands for session control", {
    x: 0.5, y: 0.8, w: 12, h: 0.3, fontSize: 12, color: DIM
  });

  const cmds = [
    ["/exit, /quit", "Exit chat session"],
    ["/help", "Show command list"],
    ["/history", "Conversation history"],
    ["/clear", "Clear conversation"],
    ["/model <name>", "Switch LLM model"],
    ["/skin <name>", "Switch TUI skin"],
    ["/tools", "List available tools"],
    ["/verbose", "Toggle verbose output"],
    ["/cron list", "List cron jobs"],
    ["/run <query>", "One-shot query"],
    ["/save <name>", "Save session"],
    ["/sessions", "List saved sessions"],
    ["/perm <action>", "Permission management"],
    ["/skills", "List installed skills"],
  ];

  // Table
  const tableData = [["Command", "Description"]];
  cmds.forEach(c => tableData.push(c));

  s.addTable(tableData, {
    x: 0.5, y: 1.3, w: 12.3,
    fontFace: "Consolas",
    fontSize: 13,
    border: { type:"solid", pt:0.5, color:BORDER },
    color: WHITE,
    fill: { color: BG2 },
    colW: [4.5, 7.8],
    rowH: 0.38,
    autoPage: false,
    margin: [4, 8, 4, 8],
  });
  // Style header row
  tableData[0].forEach((_, ci) => {
    s.addText(tableData[0][ci], {
      x: 0.5 + (ci===0?0:4.5), y: 1.3, w: ci===0?4.5:7.8, h: 0.38,
      fontSize: 12, fontFace: "Consolas", color: CYAN, bold: true,
      align: "left", valign: "middle",
      margin: [4, 8, 4, 8]
    });
  });
}

// ═══════════════════════════════════════════════════════
// SLIDE 4 — 10 Skins
// ═══════════════════════════════════════════════════════
{
  const s = pptx.addSlide();
  setBg(s, BG);
  s.addText("10 Built-in Skins", {
    x: 0.5, y: 0.25, w: 12, h: 0.55,
    fontSize: 26, fontFace: "Georgia", color: CYAN, bold: true
  });
  s.addText("Instantly switch themes via /skin <name>", {
    x: 0.5, y: 0.8, w: 12, h: 0.3, fontSize: 12, color: DIM
  });

  const skins = [
    { n:"default", c:"58A6FF", bg:"0D1117", d:"Clean default" },
    { n:"hacker", c:"00FF00", bg:"0A0A0A", d:"Matrix-inspired" },
    { n:"retro", c:"00FFFF", bg:"1A1025", d:"80s aesthetic" },
    { n:"minimal", c:"FFFFFF", bg:"0D1117", d:"Low noise" },
    { n:"ocean", c:"00BFFF", bg:"0A1628", d:"Deep blue" },
    { n:"sunset", c:"FF8C00", bg:"1A0A0A", d:"Warm palette" },
    { n:"forest", c:"66CD00", bg:"0A1A0A", d:"Natural green" },
    { n:"midnight", c:"AAAAAA", bg:"111111", d:"Dark subtle" },
    { n:"neon", c:"FF00FF", bg:"0D001A", d:"Cyberpunk" },
    { n:"mono", c:"FFFFFF", bg:"000000", d:"High contrast" },
  ];

  const cardW = 2.25;
  const cardH = 1.5;
  const gapX = 0.15;
  const gapY = 0.15;
  const startX = 0.5;
  const startY = 1.3;

  skins.forEach((sk, i) => {
    const col = i % 5;
    const row = Math.floor(i / 5);
    const cx = startX + col * (cardW + gapX);
    const cy = startY + row * (cardH + gapY);

    s.addShape(pptx.ShapeType.rect, {
      x: cx, y: cy, w: cardW, h: cardH,
      fill: { color: sk.bg }, line: { color: sk.c, width: 1.5 },
      rectRadius: 0.06
    });
    s.addText(sk.n, {
      x: cx, y: cy+0.2, w: cardW, h: 0.4,
      fontSize: 14, fontFace: "Consolas", color: sk.c, align: "center", bold: true
    });
    s.addText(sk.d, {
      x: cx, y: cy+0.65, w: cardW, h: 0.3,
      fontSize: 10, fontFace: "Calibri", color: DIM, align: "center"
    });
    // Color swatch
    s.addShape(pptx.ShapeType.rect, {
      x: cx+0.5, y: cy+1.0, w: 1.25, h: 0.25,
      fill: { color: sk.c }, rectRadius: 0.03
    });
  });
}

// ═══════════════════════════════════════════════════════
// SLIDE 5 — Architecture
// ═══════════════════════════════════════════════════════
{
  const s = pptx.addSlide();
  setBg(s, BG);
  s.addText("Architecture — 20-Layer Middleware Pipeline", {
    x: 0.5, y: 0.25, w: 12, h: 0.55,
    fontSize: 26, fontFace: "Georgia", color: CYAN, bold: true
  });
  s.addText("Dependency-ordered: each layer has a specific lifecycle hook", {
    x: 0.5, y: 0.8, w: 12, h: 0.3, fontSize: 12, color: DIM
  });

  // Pipeline visualization
  const layers = [
    "ThreadData", "Uploads", "DanglingToolCall", "SkillLoader",
    "ContextCompressor", "Todo", "Memory", "ViewImage",
    "EvidenceTracker", "ToolArgRepair", "ConfidenceScorer",
    "LLMRetry", "Logging", "SubagentLimit", "Title",
    "MemoryMiddleware", "Clarification",
    "SandboxAudit", "LoopDetection", "SecurityMiddleware"
  ];

  const lx = 0.5, ly = 1.5;
  const lw = 2.2, lh = 0.42, lg = 0.08;

  layers.forEach((name, i) => {
    const col = i % 4;
    const row = Math.floor(i / 4);
    const x = lx + col * (lw + lg);
    const y = ly + row * (lh + lg);

    // Color coding
    let color = CYAN;
    if (name.startsWith("Sandbox") || name.startsWith("Loop") || name.startsWith("Security")) color = YELLOW;
    if (name === "EvidenceTracker" || name === "ToolArgRepair" || name === "ConfidenceScorer") color = GREEN;

    s.addShape(pptx.ShapeType.rect, {
      x, y, w: lw, h: lh,
      fill: { color: BG2 }, line: { color, width: 1 },
      rectRadius: 0.04
    });
    s.addText(name, {
      x, y, w: lw, h: lh,
      fontSize: 10, fontFace: "Consolas", color, align: "center", valign: "middle"
    });
  });

  // Infrastructure / Agent Loop boxes
  const boxY = 5.2;

  s.addShape(pptx.ShapeType.rect, {
    x: 0.5, y: boxY, w: 5.8, h: 1.5, fill: { color: BG2 },
    line: { color: GREEN, width: 1 }, rectRadius: 0.05
  });
  s.addText("Agent Loop", {
    x: 0.7, y: boxY+0.1, w: 3, h: 0.35,
    fontSize: 14, fontFace: "Calibri", color: GREEN, bold: true
  });
  s.addText("ReAct pattern: think → tool_call → observe → repeat\nStatefulAgent with session persistence & interrupt/resume\nFull streaming support via chat_stream()", {
    x: 0.7, y: boxY+0.5, w: 5.4, h: 0.8,
    fontSize: 11, fontFace: "Calibri", color: DIM
  });

  s.addShape(pptx.ShapeType.rect, {
    x: 6.8, y: boxY, w: 5.8, h: 1.5, fill: { color: BG2 },
    line: { color: YELLOW, width: 1 }, rectRadius: 0.05
  });
  s.addText("Infrastructure", {
    x: 7.0, y: boxY+0.1, w: 3, h: 0.35,
    fontSize: 14, fontFace: "Calibri", color: YELLOW, bold: true
  });
  s.addText("RAG Engine · KnowledgeStore · EvidenceDB\nMemory (3 tiers) · Skills+Curator · Gateway (11 platforms)\nCredentialPool · TraceID · Prometheus Metrics", {
    x: 7.0, y: boxY+0.5, w: 5.4, h: 0.8,
    fontSize: 11, fontFace: "Calibri", color: DIM
  });
}

// ═══════════════════════════════════════════════════════
// SLIDE 6 — Quick Start
// ═══════════════════════════════════════════════════════
{
  const s = pptx.addSlide();
  setBg(s, BG);
  s.addText("Quick Start", {
    x: 0.5, y: 0.25, w: 12, h: 0.55,
    fontSize: 26, fontFace: "Georgia", color: CYAN, bold: true
  });

  // Install
  const code1 = [
    [{text:"# 安装", options:{color:DIM,fontSize:10}}],
    [{text:"pip install kairos-agent", options:{color:GREEN,fontSize:10}}],
    [{text:""}],
    [{text:"# 或开发安装", options:{color:DIM,fontSize:10}}],
    [{text:"git clone https://github.com/buer103/kairos", options:{color:GREEN,fontSize:10}}],
    [{text:"cd kairos && pip install -e .", options:{color:GREEN,fontSize:10}}],
  ];
  addTerminal(s, 0.5, 1.1, 6, 2.3, code1, {title:"Installation"});

  // Run
  const code2 = [
    [{text:"# 设置 API Key", options:{color:DIM,fontSize:10}}],
    [{text:"export DEEPSEEK_API_KEY=sk-...", options:{color:GREEN,fontSize:10}}],
    [{text:""}],
    [{text:"# 交互式聊天 (实时流式)", options:{color:DIM,fontSize:10}}],
    [{text:"kairos", options:{color:WHITE,bold:true,fontSize:11}}],
    [{text:""}],
    [{text:"# 单次查询", options:{color:DIM,fontSize:10}}],
    [{text:"kairos \"分析项目架构\"", options:{color:GREEN,fontSize:10}}],
  ];
  addTerminal(s, 6.8, 1.1, 6, 2.3, code2, {title:"Usage"});

  // Docker
  const code3 = [
    [{text:"# 启动 REST API 服务", options:{color:DIM,fontSize:10}}],
    [{text:"python -m kairos.gateway --port 8080", options:{color:GREEN,fontSize:10}}],
    [{text:""}],
    [{text:"# Docker 部署", options:{color:DIM,fontSize:10}}],
    [{text:"docker compose up -d", options:{color:GREEN,fontSize:10}}],
    [{text:""}],
    [{text:"# 健康检查", options:{color:DIM,fontSize:10}}],
    [{text:"curl http://localhost:8080/health", options:{color:WHITE,fontSize:10}}],
  ];
  addTerminal(s, 0.5, 3.7, 6, 2.3, code3, {title:"Gateway & Docker"});

  // Stats
  const stats2 = [
    ["1276 tests", "All passing"],
    ["17 providers", "DeepSeek, OpenAI, Anthropic..."],
    ["11 gateways", "Telegram, WeChat, Slack..."],
    ["6 security", "ContentRedact, Permission..."],
    ["23 tools", "Browser, MCP, Vision, RAG..."],
  ];
  const tableData2 = [["Metric", "Detail"]];
  stats2.forEach(r => tableData2.push(r));
  s.addTable(tableData2, {
    x: 6.8, y: 3.7, w: 6, fontFace: "Calibri", fontSize: 11,
    border: { type:"solid", pt:0.5, color:BORDER },
    color: WHITE, fill: { color: BG2 },
    colW: [2.2, 3.8], rowH: 0.38,
    autoPage: false, margin: [4, 8, 4, 8]
  });
  // Header
  s.addText("Metric", {
    x: 6.8, y: 3.7, w: 2.2, h: 0.38,
    fontSize: 11, fontFace: "Calibri", color: CYAN, bold: true,
    align: "left", valign: "middle", margin: [4, 8, 4, 8]
  });
  s.addText("Detail", {
    x: 9.0, y: 3.7, w: 3.8, h: 0.38,
    fontSize: 11, fontFace: "Calibri", color: CYAN, bold: true,
    align: "left", valign: "middle", margin: [4, 8, 4, 8]
  });
}

// ═══════════════════════════════════════════════════════
// SAVE
// ═══════════════════════════════════════════════════════
pptx.writeFile({ fileName: "/home/buer/workspace/github/kairos/docs/kairos-terminal-interface.pptx" })
  .then(() => console.log("PPTX created successfully"))
  .catch(err => console.error("Error:", err));
