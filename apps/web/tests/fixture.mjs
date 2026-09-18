// Controlled wire fixture only. Never imported by the application.
import { createServer } from "node:http";

let cancelled = 0;
const server = createServer(async (req, res) => {
  if (req.method === "OPTIONS") { res.writeHead(405); res.end(); return; }
  if (req.url === "/stats") { res.setHeader("Content-Type", "application/json"); res.end(JSON.stringify({ cancelled })); return; }
  if (req.url === "/health") {
    res.setHeader("Content-Type", "application/json"); res.end(JSON.stringify({ status: "healthy", worker: "up" })); return;
  }
  if (req.url !== "/v1/chat/completions") { res.writeHead(404); res.end(); return; }
  let raw = ""; for await (const chunk of req) raw += chunk;
  const payload = JSON.parse(raw);
  const prompt = payload.messages.at(-1).content;
  res.writeHead(200, { "Content-Type": "text/event-stream", "X-Request-ID": `fixture-${payload.messages.length}` });
  res.flushHeaders();
  const emit = data => res.write(`data: ${JSON.stringify(data)}\n\n`);
  if (prompt === "error") {
    emit({ error: { message: "Controlled worker failure", code: 503 } }); res.end("data: [DONE]\n\n"); return;
  }
  emit({ choices: [{ index: 0, delta: { content: "First content. " } }] });
  const timer = setTimeout(() => {
    emit({ choices: [{ index: 0, delta: { content: `History: ${payload.messages.length}.\n\n\`\`\`python\nprint('hello')\n\`\`\`` } }] });
    res.end("data: [DONE]\n\n");
  }, prompt === "abort" ? 15000 : 1800);
  res.on("close", () => { if (!res.writableEnded) cancelled++; clearTimeout(timer); });
});
server.listen(8081, "127.0.0.1", () => console.log("TEST FIXTURE on 8081; no CORS headers; no GPU"));
