export type ChatTurn = { role: "user" | "assistant"; content: string };
export type Usage = { promptTokens?: number; completionTokens?: number; totalTokens?: number };
export type StreamEvent =
  | { type: "content"; text: string }
  | { type: "usage"; usage: Usage }
  | { type: "error"; message: string; code?: number }
  | { type: "warning"; message: string }
  | { type: "done" };
export type Health = "CHECKING" | "ONLINE" | "DEGRADED" | "OFFLINE";
