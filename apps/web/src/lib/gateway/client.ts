import { SSEParser } from "./sse";
import type { ChatTurn, Health, StreamEvent } from "./types";

const base = (process.env.NEXT_PUBLIC_GATEWAY_URL || "/api/gateway").replace(/\/$/, "");
export async function checkHealth(signal: AbortSignal): Promise<Health> {
  const response = await fetch(`${base}/health`, { signal, cache: "no-store" });
  if (response.status === 503) return "DEGRADED";
  if (!response.ok) return "OFFLINE";
  const value = await response.json();
  return value.status === "healthy" && value.worker === "up" ? "ONLINE" : "DEGRADED";
}
export async function streamChat(options: {
  model: string; messages: ChatTurn[]; signal: AbortSignal;
  onHeaders: (id: string | null, accepted: boolean) => void;
  onEvent: (event: StreamEvent) => void;
}): Promise<void> {
  const response = await fetch(`${base}/v1/chat/completions`, {
    method: "POST", signal: options.signal,
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ model: options.model, messages: options.messages, stream: true }),
  });
  options.onHeaders(response.headers.get("X-Request-ID"), response.ok);
  if (!response.ok) {
    let message = `Gateway returned HTTP ${response.status}`;
    try {
      const body = await response.json();
      if (typeof body.error?.message === "string") message = body.error.message;
      else if (typeof body.detail === "string") message = body.detail;
    } catch { /* Status remains useful for non-JSON proxy failures. */ }
    options.onEvent({ type: "error", message, code: response.status });
    return;
  }
  if (!response.body || !response.headers.get("content-type")?.includes("text/event-stream")) {
    throw new Error("Gateway did not return an SSE stream.");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const parser = new SSEParser();
  let terminal = false;
  try {
    while (!terminal) {
      const { value, done } = await reader.read();
      options.signal.throwIfAborted();
      const events = parser.push(done ? decoder.decode() : decoder.decode(value, { stream: true }));
      for (const event of events) {
        options.onEvent(event);
        if (event.type === "done" || event.type === "error") terminal = true;
      }
      if (done) {
        if (!terminal) throw new Error("Stream ended before [DONE]; partial response retained.");
        break;
      }
    }
  } finally {
    await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}
