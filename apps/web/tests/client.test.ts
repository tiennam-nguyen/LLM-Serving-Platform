import { afterEach, expect, it, vi } from "vitest";
import { streamChat, checkHealth } from "../src/lib/gateway/client";
import type { StreamEvent } from "../src/lib/gateway/types";

afterEach(() => vi.unstubAllGlobals());
const options = () => ({ model: "test", messages: [{ role: "user" as const, content: "hi" }], signal: new AbortController().signal, onHeaders: vi.fn(), onEvent: vi.fn() });
it("streams UTF-8 split across bytes and preserves the response ID", async () => {
  const bytes = new TextEncoder().encode('data: {"choices":[{"delta":{"content":"你好"}}]}\n\ndata: [DONE]\n\n');
  const stream = new ReadableStream({ start(controller) {
    for (const byte of bytes) controller.enqueue(new Uint8Array([byte])); controller.close();
  } });
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(stream, { headers: { "content-type": "text/event-stream", "x-request-id": "real-id" } })));
  const opts = options(); await streamChat(opts);
  expect(opts.onHeaders).toHaveBeenCalledWith("real-id", true);
  expect(opts.onEvent.mock.calls.map(c => c[0])).toEqual([{ type: "content", text: "你好" }, { type: "done" }]);
});
it("reports HTTP errors without marking gateway acceptance", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response('{"error":{"message":"invalid"}}', { status: 400, headers: { "X-Request-ID": "bad-id" } })));
  const opts = options(); await streamChat(opts);
  expect(opts.onHeaders).toHaveBeenCalledWith("bad-id", false);
  expect(opts.onEvent).toHaveBeenCalledWith({ type: "error", message: "invalid", code: 400 });
});
it("rejects premature EOF after delivering partial content", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response('data: {"choices":[{"delta":{"content":"partial"}}]}\n\n', { headers: { "Content-Type": "text/event-stream" } })));
  const opts = options(); await expect(streamChat(opts)).rejects.toThrow("before [DONE]");
  expect(opts.onEvent).toHaveBeenCalledWith({ type: "content", text: "partial" });
});
it("passes the abort signal to fetch and cancels the reader after an abort", async () => {
  const abort = new AbortController(); const cancelled = vi.fn();
  const events: StreamEvent[] = [];
  const opts = { ...options(), signal: abort.signal, onEvent: (event: StreamEvent) => { events.push(event); abort.abort(); } };
  // Supply a second chunk so the next read observes the aborted signal.
  const fetch = vi.fn().mockResolvedValue(new Response(new ReadableStream({ start(c) {
    c.enqueue(new TextEncoder().encode('data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'));
    c.enqueue(new Uint8Array());
  }, cancel: cancelled }), { headers: { "Content-Type": "text/event-stream" } }));
  vi.stubGlobal("fetch", fetch);
  await expect(streamChat(opts)).rejects.toThrow();
  expect(events).toEqual([{ type: "content", text: "partial" }]); expect(cancelled).toHaveBeenCalled();
  expect(fetch.mock.calls[0][1].signal).toBe(abort.signal);
});
it("distinguishes healthy, unhealthy and unreachable health responses", async () => {
  const fetch = vi.fn(); vi.stubGlobal("fetch", fetch);
  fetch.mockResolvedValueOnce(new Response('{"status":"healthy","worker":"up"}'));
  expect(await checkHealth(new AbortController().signal)).toBe("ONLINE");
  fetch.mockResolvedValueOnce(new Response('{}', { status: 503 }));
  expect(await checkHealth(new AbortController().signal)).toBe("DEGRADED");
  fetch.mockRejectedValueOnce(new TypeError("Failed to fetch"));
  await expect(checkHealth(new AbortController().signal)).rejects.toThrow("Failed to fetch");
});
