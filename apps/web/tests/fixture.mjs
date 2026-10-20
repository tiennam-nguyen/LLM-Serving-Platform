// Controlled wire fixture for Playwright E2E tests (simulates FastAPI Gateway on port 8081 with Backend CORS).
import { createServer } from "node:http";

let cancelled = 0;
const traces = new Map();

function setCorsHeaders(req, res) {
  const origin = req.headers.origin || "http://127.0.0.1:3001";
  res.setHeader("Access-Control-Allow-Origin", origin);
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Accept, X-Request-ID");
  res.setHeader("Access-Control-Expose-Headers", "X-Request-ID, X-Worker-ID");
}

const server = createServer(async (req, res) => {
  setCorsHeaders(req, res);

  if (req.method === "OPTIONS") {
    res.writeHead(200);
    res.end();
    return;
  }
  if (req.url === "/stats") {
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify({ cancelled }));
    return;
  }
  if (req.url === "/health") {
    res.setHeader("Content-Type", "application/json");
    res.setHeader("X-Worker-ID", "fixture-worker-0");
    res.end(JSON.stringify({ status: "healthy", worker: "up" }));
    return;
  }
  if (req.url?.startsWith("/v1/traces/")) {
    const reqId = decodeURIComponent(req.url.slice("/v1/traces/".length));
    const trace = traces.get(reqId);
    if (!trace) {
      res.writeHead(404, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ detail: `Trace ${reqId} not found.` }));
      return;
    }
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify(trace));
    return;
  }
  if (req.url !== "/v1/chat/completions") {
    res.writeHead(404);
    res.end();
    return;
  }

  let raw = "";
  for await (const chunk of req) raw += chunk;
  const payload = JSON.parse(raw);
  const prompt = payload.messages.at(-1).content;
  const reqId = `fixture-${payload.messages.length}`;
  const arrivalIso = new Date().toISOString();

  const traceObj = {
    request_id: reqId,
    worker_id: "fixture-worker-0",
    status: "in_flight",
    milestones: {
      arrival: arrivalIso,
      routing_start: arrivalIso,
      routing_end: arrivalIso,
      worker_start: arrivalIso,
      first_token: arrivalIso,
      last_token: null,
    },
    metrics_ms: { routing_ms: 0.1, ttft_ms: 12.5, tpot_ms: 4.2, e2e_ms: null },
  };
  traces.set(reqId, traceObj);

  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "X-Request-ID": reqId,
    "X-Worker-ID": "fixture-worker-0",
    "Access-Control-Allow-Origin": req.headers.origin || "http://127.0.0.1:3001",
    "Access-Control-Expose-Headers": "X-Request-ID, X-Worker-ID",
  });
  res.flushHeaders();

  const emit = (data) => res.write(`data: ${JSON.stringify(data)}\n\n`);
  if (prompt === "error") {
    traceObj.status = "error";
    traceObj.milestones.last_token = new Date().toISOString();
    traceObj.metrics_ms.e2e_ms = 15.0;
    emit({ error: { message: "Controlled worker failure", code: 503 } });
    res.end("data: [DONE]\n\n");
    return;
  }

  emit({ choices: [{ index: 0, delta: { content: "First content. " } }] });
  const timer = setTimeout(
    () => {
      traceObj.status = "completed";
      traceObj.milestones.last_token = new Date().toISOString();
      traceObj.metrics_ms.e2e_ms = 1812.5;
      emit({
        choices: [
          {
            index: 0,
            delta: {
              content: `History: ${payload.messages.length}.\n\n\`\`\`python\nprint('hello')\n\`\`\``,
            },
          },
        ],
      });
      res.end("data: [DONE]\n\n");
    },
    prompt === "abort" ? 15000 : 1800,
  );

  res.on("close", () => {
    if (!res.writableEnded) {
      cancelled++;
      traceObj.status = "cancelled";
      traceObj.milestones.last_token = new Date().toISOString();
      traceObj.metrics_ms.e2e_ms = 120.0;
    }
    clearTimeout(timer);
  });
});

server.listen(8081, "127.0.0.1", () =>
  console.log("TEST FIXTURE on 8081; Backend CORS enabled; no GPU"),
);
