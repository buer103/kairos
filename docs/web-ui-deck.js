const pptxgen = require("pptxgenjs");
const pptx = new pptxgen();

const BG = "08090A", PANEL = "0F1011", SURFACE = "191A1B";
const TEXT = "F7F8F8", TEXT2 = "D0D6E0", TEXT3 = "8A8F98", TEXT4 = "62666D";
const ACCENT = "5E6AD2", BORDER = "3E3E44", GREEN = "27A644";

pptx.layout = "LAYOUT_WIDE";
pptx.author = "Kairos";
pptx.title = "Kairos Web UI";

const fontH = "Georgia", fontB = "Calibri";

function ds(title) {
  const s = pptx.addSlide();
  s.background = { color: BG };
  s.addText("KAIROS", { x: 0.6, y: 0.25, w: 2, h: 0.4, fontSize: 10, color: ACCENT, fontFace: fontB, bold: true, charSpacing: 3 });
  s.addText(title, { x: 0.6, y: 0.65, w: 9, h: 0.6, fontSize: 28, color: TEXT, fontFace: fontH, bold: true });
  return s;
}
function panel(s, x, y, w, h, c) {
  s.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: c || SURFACE }, rectRadius: 0.08, line: { color: BORDER, width: 0.5 } });
}

// SLIDE 1 — Title
{
  const s = pptx.addSlide();
  s.background = { color: BG };
  s.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: 13.333, h: 3.2, fill: { color: ACCENT } });
  s.addShape(pptx.shapes.OVAL, { x: 1.2, y: 1.0, w: 0.3, h: 0.3, fill: { color: GREEN } });
  s.addText("Kairos Web UI", { x: 1.2, y: 1.3, w: 10, h: 1.2, fontSize: 48, color: "FFFFFF", fontFace: fontH, bold: true });
  s.addText("Linear-inspired dark theme  ·  SSE streaming  ·  Session management", { x: 1.2, y: 2.5, w: 10, h: 0.5, fontSize: 16, color: "C8D0FF", fontFace: fontB });

  const feats = [
    ["Chat", "Real-time SSE\nstreaming output"],
    ["Tools", "Collapsible tool\ncall details"],
    ["Sessions", "List, load, save,\ndelete sessions"],
    ["Mobile", "Responsive design\nworks everywhere"],
  ];
  feats.forEach((f, i) => {
    const x = 1.2 + i * 2.8;
    panel(s, x, 3.8, 2.5, 1.8);
    s.addText(f[0], { x: x+0.2, y: 3.95, w: 2.1, h: 0.4, fontSize: 15, color: TEXT, fontFace: fontH, bold: true });
    s.addText(f[1], { x: x+0.2, y: 4.4, w: 2.1, h: 1.0, fontSize: 12, color: TEXT3, fontFace: fontB, lineSpacing: 18 });
  });
  s.addText("kairos web  →  http://127.0.0.1:8080", { x: 1.2, y: 6.0, w: 8, h: 0.4, fontSize: 14, color: ACCENT, fontFace: "Consolas", bold: true });
}

// SLIDE 2 — Chat Interface
{
  const s = ds("Chat Interface — SSE Streaming");

  // Browser frame
  s.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: 1.6, w: 12.3, h: 5.4, fill: { color: PANEL }, rectRadius: 0.12, line: { color: BORDER, width: 0.5 } });
  s.addShape(pptx.shapes.RECTANGLE, { x: 0.5, y: 1.6, w: 12.3, h: 0.4, fill: { color: "141516" } });
  s.addShape(pptx.shapes.OVAL, { x: 0.7, y: 1.72, w: 0.14, h: 0.14, fill: { color: "E5484D" } });
  s.addShape(pptx.shapes.OVAL, { x: 0.95, y: 1.72, w: 0.14, h: 0.14, fill: { color: "F5A623" } });
  s.addShape(pptx.shapes.OVAL, { x: 1.2, y: 1.72, w: 0.14, h: 0.14, fill: { color: GREEN } });
  s.addText("localhost:8080", { x: 4.5, y: 1.62, w: 4, h: 0.35, fontSize: 9, color: TEXT3, fontFace: fontB, align: "center" });

  // Sidebar
  s.addShape(pptx.shapes.RECTANGLE, { x: 0.5, y: 2.0, w: 2.5, h: 5.0, fill: { color: PANEL }, line: { color: BORDER, width: 0.3 } });
  s.addText("🔵 Kairos", { x: 0.7, y: 2.1, w: 2, h: 0.3, fontSize: 13, color: TEXT, fontFace: fontH, bold: true });
  ["debug-431", "api-design", "refactor", "new-feature"].forEach((n, i) => {
    const y = 2.6 + i * 0.45;
    s.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: 0.65, y, w: 2.15, h: 0.35, fill: { color: i===0 ? "1E2152" : "0F1011" }, rectRadius: 0.04 });
    s.addText(n, { x: 0.75, y, w: 1.8, h: 0.35, fontSize: 10, color: i===0 ? ACCENT : TEXT3, fontFace: fontB, valign: "middle" });
  });

  // Chat header
  s.addText("Web UI", { x: 3.2, y: 2.1, w: 3, h: 0.3, fontSize: 11, color: TEXT3, fontFace: fontB });
  s.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: 11.0, y: 2.08, w: 0.9, h: 0.3, fill: { color: SURFACE }, rectRadius: 999, line: { color: BORDER, width: 0.3 } });
  s.addText("kairos", { x: 11.0, y: 2.08, w: 0.9, h: 0.3, fontSize: 9, color: TEXT2, fontFace: fontB, align: "center", valign: "middle" });

  // User message
  s.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: 8.2, y: 2.6, w: 4.3, h: 0.7, fill: { color: SURFACE }, rectRadius: 0.08, line: { color: BORDER, width: 0.3 } });
  s.addText("Build a health check\nendpoint for Kairos", { x: 8.4, y: 2.68, w: 3.9, h: 0.55, fontSize: 11, color: TEXT2, fontFace: fontB, lineSpacing: 16 });

  // Agent reply
  s.addText("KAIROS", { x: 3.3, y: 3.5, w: 1, h: 0.2, fontSize: 8, color: ACCENT, fontFace: fontB });
  s.addText("Here is a FastAPI health check endpoint:", { x: 3.3, y: 3.72, w: 8, h: 0.3, fontSize: 12, color: TEXT2, fontFace: fontB });

  // Code
  s.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: 3.3, y: 4.1, w: 8.5, h: 1.8, fill: { color: SURFACE }, rectRadius: 0.08, line: { color: BORDER, width: 0.3 } });
  s.addText("@app.get(\"/health\")\nasync def health():\n    return {\"status\": \"ok\"}", { x: 3.5, y: 4.2, w: 8.0, h: 1.5, fontSize: 10, color: TEXT2, fontFace: "Consolas", lineSpacing: 18 });

  // Input
  s.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: 3.2, y: 6.4, w: 8.6, h: 0.42, fill: { color: SURFACE }, rectRadius: 0.04, line: { color: BORDER, width: 0.3 } });
  s.addText("Type a message...", { x: 3.4, y: 6.4, w: 4, h: 0.42, fontSize: 11, color: TEXT4, fontFace: fontB, valign: "middle" });
  s.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: 11.4, y: 6.4, w: 0.5, h: 0.42, fill: { color: ACCENT }, rectRadius: 0.04 });
  s.addText("→", { x: 11.4, y: 6.4, w: 0.5, h: 0.42, fontSize: 14, color: "FFFFFF", fontFace: fontB, align: "center", valign: "middle" });
}

// SLIDE 3 — Tool Calls
{
  const s = ds("Tool Call Visibility");

  s.addText("Every tool call shown inline — click to expand details", { x: 0.6, y: 1.5, w: 10, h: 0.4, fontSize: 13, color: TEXT3, fontFace: fontB });
  s.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 2.2, w: 12.0, h: 4.6, fill: { color: PANEL }, rectRadius: 0.1, line: { color: BORDER, width: 0.3 } });

  s.addText("KAIROS", { x: 0.9, y: 2.35, w: 1, h: 0.2, fontSize: 8, color: ACCENT, fontFace: fontB });
  s.addText("Let me check the code and run tests.", { x: 0.9, y: 2.6, w: 8, h: 0.3, fontSize: 12, color: TEXT2, fontFace: fontB });

  // Tool 1 — collapsed
  s.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: 1.0, y: 3.1, w: 7, h: 0.35, fill: { color: SURFACE }, rectRadius: 0.04, line: { color: BORDER, width: 0.3 } });
  s.addText("🔧 search_files", { x: 1.15, y: 3.1, w: 3, h: 0.35, fontSize: 10, color: "F5A623", fontFace: fontB, bold: true, valign: "middle" });
  s.addText("3 matches found", { x: 4.5, y: 3.1, w: 3, h: 0.35, fontSize: 10, color: TEXT4, fontFace: "Consolas", valign: "middle" });

  // Tool 2 — expanded
  s.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: 1.0, y: 3.55, w: 7, h: 1.3, fill: { color: SURFACE }, rectRadius: 0.04, line: { color: ACCENT, width: 0.5 } });
  s.addText("🔧 terminal", { x: 1.15, y: 3.55, w: 3, h: 0.35, fontSize: 10, color: "F5A623", fontFace: fontB, bold: true, valign: "middle" });
  s.addText("✓ 1344 passed in 12.3s", { x: 4.5, y: 3.55, w: 3, h: 0.35, fontSize: 10, color: GREEN, fontFace: "Consolas", valign: "middle" });
  s.addShape(pptx.shapes.LINE, { x: 1.0, y: 3.9, w: 7, h: 0, line: { color: BORDER, width: 0.3 } });
  s.addText("Arguments:", { x: 1.15, y: 3.95, w: 3, h: 0.25, fontSize: 9, color: TEXT3, fontFace: fontB });
  s.addText("{ \"command\": \"pytest -q\", \"timeout\": 120 }", { x: 1.15, y: 4.2, w: 6, h: 0.3, fontSize: 9, color: TEXT2, fontFace: "Consolas" });
  s.addText("Duration: 12,340ms", { x: 1.15, y: 4.55, w: 3, h: 0.2, fontSize: 9, color: TEXT4, fontFace: fontB });

  // Tool 3
  s.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: 1.0, y: 5.0, w: 7, h: 0.35, fill: { color: SURFACE }, rectRadius: 0.04, line: { color: BORDER, width: 0.3 } });
  s.addText("🔧 read_file", { x: 1.15, y: 5.0, w: 3, h: 0.35, fontSize: 10, color: "F5A623", fontFace: fontB, bold: true, valign: "middle" });
  s.addText("45 lines", { x: 4.5, y: 5.0, w: 2, h: 0.35, fontSize: 10, color: TEXT4, fontFace: "Consolas", valign: "middle" });

  s.addText("All 1,344 tests pass. Kairos is healthy.", { x: 0.9, y: 5.6, w: 10, h: 0.3, fontSize: 12, color: TEXT2, fontFace: fontB });
}

// SLIDE 4 — Sessions
{
  const s = ds("Session Management");
  s.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 1.6, w: 12.0, h: 5.2, fill: { color: PANEL }, rectRadius: 0.1, line: { color: BORDER, width: 0.3 } });

  // Sidebar
  s.addShape(pptx.shapes.RECTANGLE, { x: 0.7, y: 1.7, w: 2.8, h: 5.0, fill: { color: "0A0A0D" }, line: { color: BORDER, width: 0.3 } });
  s.addText("🔵 Kairos", { x: 0.9, y: 1.85, w: 2, h: 0.3, fontSize: 14, color: TEXT, fontFace: fontH, bold: true });

  [["debug-431","8 turns"],["api-design","24 turns"],["refactor-v2","15 turns"],["rag-tuning","32 turns"]].forEach((sess, i) => {
    const y = 2.4 + i * 0.55;
    s.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: 0.8, y, w: 2.5, h: 0.45, fill: { color: i===0 ? "1E2152" : "0F1011" }, rectRadius: 0.04 });
    s.addText(sess[0], { x: 0.95, y, w: 1.5, h: 0.28, fontSize: 11, color: i===0 ? ACCENT : TEXT2, fontFace: fontB, bold: true });
    s.addText(sess[1], { x: 0.95, y: y+0.22, w: 2, h: 0.2, fontSize: 9, color: TEXT4, fontFace: fontB });
  });

  s.addText("Operations", { x: 3.9, y: 2.0, w: 3, h: 0.3, fontSize: 18, color: TEXT, fontFace: fontH, bold: true });

  [["List","GET /api/sessions"],["Save","POST /api/sessions/save"],["Load","POST /api/sessions/load"],["Delete","DELETE /api/sessions/:name"]].forEach((op, i) => {
    const y = 2.55 + i * 0.95;
    s.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: 3.9, y, w: 8.3, h: 0.8, fill: { color: SURFACE }, rectRadius: 0.06, line: { color: BORDER, width: 0.3 } });
    s.addText(op[0], { x: 4.1, y: y+0.05, w: 2, h: 0.3, fontSize: 14, color: TEXT, fontFace: fontH, bold: true });
    s.addText(op[1], { x: 4.1, y: y+0.38, w: 7, h: 0.35, fontSize: 10, color: ACCENT, fontFace: "Consolas" });
  });
}

// SLIDE 5 — Mobile
{
  const s = ds("Mobile Responsive");

  // Phone frame
  const px = 4.5, py = 1.3;
  s.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: px-0.15, y: py-0.15, w: 3.5, h: 5.5, fill: { color: "1A1A1E" }, rectRadius: 0.3, line: { color: BORDER, width: 1 } });
  s.addShape(pptx.shapes.RECTANGLE, { x: px, y: py, w: 3.2, h: 5.2, fill: { color: BG } });

  s.addShape(pptx.shapes.RECTANGLE, { x: px, y: py, w: 3.2, h: 0.35, fill: { color: PANEL } });
  s.addText("Kairos", { x: px+0.15, y: py+0.02, w: 1.5, h: 0.3, fontSize: 11, color: TEXT, fontFace: fontH, bold: true });

  s.addText("KAIROS", { x: px+0.15, y: py+0.55, w: 1, h: 0.15, fontSize: 6, color: ACCENT, fontFace: fontB });
  s.addText("Welcome to\nKairos Web UI.", { x: px+0.15, y: py+0.7, w: 2.5, h: 0.4, fontSize: 10, color: TEXT2, fontFace: fontB, lineSpacing: 14 });

  s.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: px+1.8, y: py+1.2, w: 1.3, h: 0.55, fill: { color: SURFACE }, rectRadius: 0.06, line: { color: BORDER, width: 0.2 } });
  s.addText("What tools?", { x: px+1.9, y: py+1.25, w: 1.1, h: 0.45, fontSize: 9, color: TEXT2, fontFace: fontB });

  s.addText("KAIROS", { x: px+0.15, y: py+2.0, w: 1, h: 0.15, fontSize: 6, color: ACCENT, fontFace: fontB });
  s.addText("24 tools: file,\nterminal, web,\nbrowser, RAG...", { x: px+0.15, y: py+2.15, w: 2.8, h: 0.7, fontSize: 9, color: TEXT2, fontFace: fontB, lineSpacing: 14 });

  s.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: px+0.2, y: py+3.0, w: 2.8, h: 0.3, fill: { color: SURFACE }, rectRadius: 0.04, line: { color: BORDER, width: 0.2 } });
  s.addText("🔧 search_files", { x: px+0.35, y: py+3.0, w: 2.5, h: 0.3, fontSize: 8, color: "F5A623", fontFace: fontB, valign: "middle" });

  s.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: px, y: py+4.65, w: 2.8, h: 0.35, fill: { color: SURFACE }, rectRadius: 0.04, line: { color: BORDER, width: 0.2 } });
  s.addText("Ask anything...", { x: px+0.15, y: py+4.65, w: 2, h: 0.35, fontSize: 9, color: TEXT4, fontFace: fontB, valign: "middle" });
  s.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: px+2.8, y: py+4.65, w: 0.35, h: 0.35, fill: { color: ACCENT }, rectRadius: 0.04 });
  s.addText("→", { x: px+2.8, y: py+4.65, w: 0.35, h: 0.35, fontSize: 12, color: "FFFFFF", fontFace: fontB, align: "center", valign: "middle" });

  // Feature list
  s.addText("Responsive Features", { x: 9.0, y: 1.5, w: 3, h: 0.4, fontSize: 18, color: TEXT, fontFace: fontH, bold: true });
  ["Sidebar hides at 768px","Full-width chat on mobile","Tap to expand tool calls","42px touch targets"].forEach((f, i) => {
    const y = 2.2 + i * 0.55;
    s.addShape(pptx.shapes.OVAL, { x: 9.0, y: y+0.04, w: 0.16, h: 0.16, fill: { color: ACCENT } });
    s.addText(f, { x: 9.3, y, w: 3, h: 0.3, fontSize: 12, color: TEXT2, fontFace: fontB, valign: "middle" });
  });

  s.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: 9.0, y: 5.6, w: 3.5, h: 0.8, fill: { color: SURFACE }, rectRadius: 0.06, line: { color: BORDER, width: 0.3 } });
  s.addText("Zero Dependencies", { x: 9.2, y: 5.68, w: 3, h: 0.3, fontSize: 13, color: TEXT, fontFace: fontH, bold: true });
  s.addText("Single HTML · No npm · No build\nWorks on any modern browser", { x: 9.2, y: 5.95, w: 3.1, h: 0.4, fontSize: 10, color: TEXT3, fontFace: fontB, lineSpacing: 14 });
}

// SLIDE 6 — Launch
{
  const s = pptx.addSlide();
  s.background = { color: ACCENT };
  s.addText("KAIROS", { x: 1.2, y: 1.0, w: 3, h: 0.4, fontSize: 10, color: "C8D0FF", fontFace: fontB, bold: true, charSpacing: 3 });
  s.addText("Launch the Web UI", { x: 1.2, y: 1.5, w: 10, h: 1.0, fontSize: 40, color: "FFFFFF", fontFace: fontH, bold: true });

  s.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: 1.2, y: 2.8, w: 10.5, h: 1.8, fill: { color: "3E3E44" }, rectRadius: 0.1 });
  s.addText("$ kairos web", { x: 1.5, y: 3.0, w: 4, h: 0.5, fontSize: 28, color: "FFFFFF", fontFace: "Consolas", bold: true });
  s.addText("✨ Kairos Web UI: http://127.0.0.1:8080", { x: 1.5, y: 3.55, w: 8, h: 0.4, fontSize: 16, color: "C8D0FF", fontFace: "Consolas" });
  s.addText("Open your browser and start chatting", { x: 1.5, y: 4.1, w: 5, h: 0.3, fontSize: 14, color: "C8D0FF", fontFace: fontB });

  [["811 LOC","Server + SPA"],["0 new deps","Pure aiohttp"],["1,300+ tests","All passing"],["Linear theme","Dark mode"]].forEach((st, i) => {
    s.addText(st[0], { x: 1.2+i*2.8, y: 5.2, w: 2.5, h: 0.5, fontSize: 22, color: "FFFFFF", fontFace: fontH, bold: true });
    s.addText(st[1], { x: 1.2+i*2.8, y: 5.7, w: 2.5, h: 0.3, fontSize: 12, color: "C8D0FF", fontFace: fontB });
  });

  s.addText("github.com/buer103/kairos", { x: 1.2, y: 6.5, w: 5, h: 0.4, fontSize: 14, color: "C8D0FF", fontFace: "Consolas" });
}

pptx.writeFile({ fileName: "/home/buer/workspace/github/kairos/docs/kairos-web-ui.pptx" })
  .then(() => console.log("OK"))
  .catch(e => console.error(e));
