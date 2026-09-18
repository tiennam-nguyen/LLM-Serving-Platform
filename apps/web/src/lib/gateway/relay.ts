export function sameOrigin(request: Request): boolean {
  const origin = request.headers.get("origin");
  if (!origin) return true;
  try {
    const source = new URL(origin);
    const target = new URL(request.url);
    // Next normalizes loopback request URLs to localhost; Host preserves browser authority.
    return source.host === (request.headers.get("host") || target.host) && source.protocol === target.protocol;
  } catch { return false; }
}

/** Fixed-route server relay. The upstream address comes only from server configuration. */
export async function relay(request: Request, path: "/health" | "/v1/chat/completions") {
  if (request.method === "POST" && !sameOrigin(request)) {
    return Response.json({ error: { message: "Cross-origin submissions are not allowed." } }, { status: 403 });
  }
  const abort = new AbortController();
  const disconnect = () => abort.abort();
  request.signal.addEventListener("abort", disconnect, { once: true });
  if (request.signal.aborted) abort.abort();
  const timeout = setTimeout(() => abort.abort(), 10000);
  function cleanup() { clearTimeout(timeout); request.signal.removeEventListener("abort", disconnect); }
  try {
    const base = new URL(process.env.GATEWAY_URL || "http://127.0.0.1:8080");
    if (!["http:", "https:"].includes(base.protocol) || base.username || base.password) throw new Error("Invalid gateway configuration");
    const upstream = await fetch(new URL(path, base), {
      method: request.method, signal: abort.signal, cache: "no-store",
      headers: { Accept: path === "/health" ? "application/json" : "text/event-stream", "Content-Type": "application/json" },
      body: request.method === "POST" ? await request.text() : undefined,
    });
    clearTimeout(timeout);
    const headers = new Headers({
      "Content-Type": upstream.headers.get("Content-Type") || "application/json",
      "Cache-Control": "no-store, no-transform", "X-Accel-Buffering": "no",
    });
    const id = upstream.headers.get("X-Request-ID");
    if (id) headers.set("X-Request-ID", id);
    if (!upstream.body) { cleanup(); return new Response(null, { status: upstream.status, headers }); }
    const reader = upstream.body.getReader();
    const body = new ReadableStream<Uint8Array>({
      async pull(controller) {
        try {
          const { value, done } = await reader.read();
          if (done) { cleanup(); reader.releaseLock(); controller.close(); }
          else controller.enqueue(value);
        } catch (error) { cleanup(); controller.error(error); }
      },
      async cancel() { abort.abort(); cleanup(); await reader.cancel().catch(() => undefined); },
    });
    return new Response(body, { status: upstream.status, headers });
  } catch {
    cleanup();
    return Response.json({ error: { message: "Gateway unreachable or response headers timed out. Check GATEWAY_URL and gateway availability." } }, { status: 502 });
  }
}
