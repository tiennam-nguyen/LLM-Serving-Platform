import { describe, expect, it } from "vitest";
import { SSEParser } from "../src/lib/gateway/sse";

const content = 'data: {"choices":[{"index":0,"delta":{"content":"hello"}}]}\n\n';
describe("SSEParser", () => {
  it("decodes one event and does not equate it with a token", () => {
    expect(new SSEParser().push(content)).toEqual([{ type: "content", text: "hello" }]);
  });
  it("handles every possible two-chunk split including CRLF boundaries", () => {
    const input = content.replaceAll("\n", "\r\n");
    for (let at = 0; at <= input.length; at++) {
      const parser = new SSEParser();
      expect([...parser.push(input.slice(0, at)), ...parser.push(input.slice(at))]).toEqual([{ type: "content", text: "hello" }]);
    }
  });
  it("handles multiple events and ignores records after DONE", () => {
    expect(new SSEParser().push(content + content + 'data: [DONE]\n\n' + content)).toEqual([
      { type: "content", text: "hello" }, { type: "content", text: "hello" }, { type: "done" },
    ]);
  });
  it("keeps error terminal even when followed by DONE", () => {
    expect(new SSEParser().push('data: {"error":{"message":"worker down","code":503}}\n\ndata: [DONE]\n\n')).toEqual([
      { type: "error", message: "worker down", code: 503 },
    ]);
  });
  it("supports named errors", () => {
    expect(new SSEParser().push('event: error\ndata: timeout\n\n')).toEqual([{ type: "error", message: "timeout" }]);
  });
  it("survives malformed JSON, primitives, comments and empty lines", () => {
    const events = new SSEParser().push('\n\n: keepalive\n\ndata: nope\n\ndata: null\n\n' + content);
    expect(events.map(e => e.type)).toEqual(["warning", "warning", "content"]);
  });
  it("joins multiline data and records only valid explicit usage", () => {
    expect(new SSEParser().push('data: {"usage":\ndata: {"prompt_tokens":0,"completion_tokens":-1,"total_tokens":4}}\n\n')).toEqual([
      { type: "usage", usage: { promptTokens: 0, totalTokens: 4 } },
    ]);
  });
  it("waits for blank-line delimiter", () => {
    expect(new SSEParser().push(content.trim())).toEqual([]);
  });
});
