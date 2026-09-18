import { afterEach, expect, it, vi } from "vitest";
import { relay, sameOrigin } from "../src/lib/gateway/relay";

afterEach(() => vi.unstubAllGlobals());
it("accepts the actual Host despite Next loopback URL normalization", () => {
  expect(sameOrigin(new Request("http://localhost:3001/api/gateway/v1/chat/completions", {
    headers: { host: "127.0.0.1:3001", origin: "http://127.0.0.1:3001" },
  }))).toBe(true);
});
it.each(["http://other.example:3001", "http://127.0.0.1:9000", "https://127.0.0.1:3001", "null"])("rejects mismatched origin %s", origin => {
  expect(sameOrigin(new Request("http://localhost:3001/", { headers: { host: "127.0.0.1:3001", origin } }))).toBe(false);
});
it("does not contact upstream on a cross-origin POST", async () => {
  const fetch = vi.fn(); vi.stubGlobal("fetch", fetch);
  const result = await relay(new Request("http://localhost:3001/", { method: "POST", headers: { origin: "https://other.example" } }), "/v1/chat/completions");
  expect(result.status).toBe(403); expect(fetch).not.toHaveBeenCalled();
});
