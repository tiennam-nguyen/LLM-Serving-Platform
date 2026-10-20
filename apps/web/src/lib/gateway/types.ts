export type ChatTurn = { role: "user" | "assistant"; content: string };
export type Usage = { promptTokens?: number; completionTokens?: number; totalTokens?: number };
export type ServerTrace = {
  request_id: string;
  worker_id: string;
  status: string;
  milestones: {
    arrival: string | null;
    routing_start: string | null;
    routing_end: string | null;
    worker_start: string | null;
    first_token: string | null;
    last_token: string | null;
  };
  metrics_ms: {
    routing_ms: number | null;
    ttft_ms: number | null;
    tpot_ms: number | null;
    e2e_ms: number | null;
  };
};
export type StreamEvent =
  | { type: "content"; text: string }
  | { type: "usage"; usage: Usage }
  | { type: "error"; message: string; code?: number }
  | { type: "warning"; message: string }
  | { type: "done" };
export type Health = "CHECKING" | "ONLINE" | "DEGRADED" | "OFFLINE";
