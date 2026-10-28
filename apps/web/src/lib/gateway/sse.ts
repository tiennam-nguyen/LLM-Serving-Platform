import type { StreamEvent, Usage } from "./types";

function object(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown> : undefined;
}

function decode(data: string, name: string): StreamEvent[] {
  if (data.trim() === "[DONE]") return [{ type: "done" }];
  let payload: Record<string, unknown> | undefined;
  try { payload = object(JSON.parse(data)); } catch {
    return name === "error" ? [{ type: "error", message: data || "Gateway stream error" }]
      : [{ type: "warning", message: "Malformed SSE JSON skipped; response may be incomplete." }];
  }
  if (!payload) return [{ type: "warning", message: "Invalid SSE payload skipped." }];
  const error = object(payload.error);
  if (payload.error || name === "error") return [{
    type: "error",
    message: typeof error?.message === "string" ? error.message
      : typeof payload.message === "string" ? payload.message : "Gateway stream error",
    code: typeof error?.code === "number" ? error.code : undefined,
  }];
  const events: StreamEvent[] = [];
  const choices = Array.isArray(payload.choices) ? payload.choices : [];
  const choice = object(choices.find(c => object(c)?.index === 0) ?? choices[0]);
  const content = object(choice?.delta)?.content;
  if (typeof content === "string" && content.length) events.push({ type: "content", text: content });
  const reported = object(payload.usage);
  if (reported) {
    const usage: Usage = {};
    for (const [wire, local] of [["prompt_tokens", "promptTokens"], ["completion_tokens", "completionTokens"], ["total_tokens", "totalTokens"]] as const) {
      const count = reported[wire];
      if (typeof count === "number" && Number.isSafeInteger(count) && count >= 0) usage[local] = count;
    }
    if (Object.keys(usage).length) events.push({ type: "usage", usage });
  }
  return events;
}

/** Stateful line decoder; CRLF may itself straddle network chunks. */
export class SSEParser {
  private line = "";
  private data: string[] = [];
  private name = "";
  private afterCR = false;
  private ended = false;

  push(chunk: string): StreamEvent[] {
    const output: StreamEvent[] = [];
    for (const char of chunk) {
      if (this.ended) break;
      if (this.afterCR && char === "\n") { this.afterCR = false; continue; }
      this.afterCR = false;
      if (char === "\r" || char === "\n") {
        this.afterCR = char === "\r";
        if (!this.line) {
          if (this.data.length) {
            const events = decode(this.data.join("\n"), this.name);
            output.push(...events);
            this.ended = events.some(e => e.type === "done" || e.type === "error");
          }
          this.data = []; this.name = "";
        } else {
          const colon = this.line.indexOf(":");
          const field = colon < 0 ? this.line : this.line.slice(0, colon);
          const value = colon < 0 ? "" : this.line.slice(colon + 1).replace(/^ /, "");
          if (field === "data") this.data.push(value);
          if (field === "event") this.name = value;
        }
        this.line = "";
      } else this.line += char;
    }
    return output;
  }
}
