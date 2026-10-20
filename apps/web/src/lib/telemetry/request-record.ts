import type { ChatTurn, ServerTrace, StreamEvent, Usage } from "../gateway/types";

export type RequestStatus = "submitting" | "streaming" | "completed" | "failed" | "cancelled";
export type RequestRecord = {
  localId: string; sequence: number; requestId: string | null; workerId?: string | null;
  model: string; status: RequestStatus; submittedAt: number; acceptedAt?: number;
  firstContentAt?: number; completedAt?: number; usage?: Usage; serverTrace?: ServerTrace;
  error?: { code?: number; message: string }; warnings: string[];
  content: string; history: ChatTurn[];
};
export const isRunning = (r: RequestRecord) => r.status === "submitting" || r.status === "streaming";
export function transition(record: RequestRecord, event: StreamEvent, now: number): RequestRecord {
  if (!isRunning(record)) return record;
  switch (event.type) {
    case "content": return event.text ? { ...record, status: "streaming", content: record.content + event.text, firstContentAt: record.firstContentAt ?? now } : record;
    case "usage": return { ...record, usage: { ...record.usage, ...event.usage } };
    case "warning": return { ...record, warnings: [...new Set([...record.warnings, event.message])].slice(0, 5) };
    case "error": return { ...record, status: "failed", error: { message: event.message, code: event.code }, completedAt: now };
    case "done": return { ...record, status: "completed", completedAt: now };
  }
}
export function cancel(record: RequestRecord, now: number): RequestRecord {
  return isRunning(record) ? { ...record, status: "cancelled", completedAt: now } : record;
}
export const milliseconds = (start?: number, end?: number) =>
  start === undefined || end === undefined ? "—" : `${Math.max(0, end - start).toFixed(0)} ms`;
