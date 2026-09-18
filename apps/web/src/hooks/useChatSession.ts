"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { checkHealth, streamChat } from "../lib/gateway/client";
import type { ChatTurn, Health } from "../lib/gateway/types";
import { cancel, isRunning, transition, type RequestRecord } from "../lib/telemetry/request-record";

export type FeedMessage = { id: string; role: "user"; content: string } | { id: string; role: "assistant"; request: RequestRecord };
const SESSION_LIMIT = 50;

export function useChatSession(model: string) {
  const [messages, setMessages] = useState<FeedMessage[]>([]);
  const [selected, select] = useState<string | null>(null);
  const [health, setHealth] = useState<Health>("CHECKING");
  const [active, setActive] = useState(false);
  const controller = useRef<AbortController | null>(null);
  const sequence = useRef(0);
  const healthController = useRef<AbortController | null>(null);
  const refreshHealth = useCallback(async () => {
    healthController.current?.abort();
    const check = new AbortController();
    healthController.current = check;
    const timeout = setTimeout(() => check.abort(), 8000);
    try { setHealth(await checkHealth(check.signal)); }
    catch { if (healthController.current === check) setHealth("OFFLINE"); }
    finally { clearTimeout(timeout); }
  }, []);
  useEffect(() => {
    const initial = setTimeout(() => void refreshHealth(), 0);
    const interval = setInterval(() => void refreshHealth(), 30000);
    return () => {
      clearTimeout(initial); clearInterval(interval);
      healthController.current?.abort(); controller.current?.abort();
    };
  }, [refreshHealth]);

  function update(id: string, change: (record: RequestRecord) => RequestRecord) {
    setMessages(current => current.map(message => message.role === "assistant" && message.id === id
      ? { ...message, request: change(message.request) } : message));
  }

  async function generate(history: ChatTurn[], targetModel: string, user?: string) {
    if (controller.current || !targetModel.trim()) return;
    const abort = new AbortController();
    controller.current = abort;
    setActive(true);
    const id = crypto.randomUUID();
    const request: RequestRecord = {
      localId: id, sequence: ++sequence.current, requestId: null, model: targetModel,
      status: "submitting", submittedAt: performance.now(), content: "", history, warnings: [],
    };
    setMessages(current => {
      const next: FeedMessage[] = [...current];
      if (user !== undefined) next.push({ id: crypto.randomUUID(), role: "user", content: user });
      next.push({ id, role: "assistant", request });
      // Bound feed and records together so every visible response stays inspectable.
      while (next.filter(m => m.role === "assistant").length > SESSION_LIMIT) {
        const oldest = next.findIndex(m => m.role === "assistant");
        next.splice(0, oldest + 1);
      }
      return next;
    });
    select(id);
    try {
      await streamChat({ model: targetModel, messages: history, signal: abort.signal,
        onHeaders: (requestId, accepted) => {
          const now = performance.now();
          update(id, r => ({ ...r, requestId, acceptedAt: accepted ? now : undefined }));
        },
        onEvent: event => {
          const now = performance.now();
          update(id, r => transition(r, event, now));
          if (event.type === "error") void refreshHealth();
        },
      });
    } catch (error) {
      const now = performance.now();
      update(id, r => abort.signal.aborted ? cancel(r, now) : transition(r, {
        type: "error", message: error instanceof Error ? error.message : "Request failed",
      }, now));
      if (!abort.signal.aborted) void refreshHealth();
    } finally {
      if (controller.current === abort) { controller.current = null; setActive(false); }
    }
  }

  function send(text: string) {
    if (!text.trim() || active || !model) return;
    const last = [...messages].reverse().find(m => m.role === "assistant");
    // Continue the latest branch. Failed/aborted text remains visible but is not fed back.
    const history: ChatTurn[] = last?.role === "assistant"
      ? [...last.request.history, ...(last.request.status === "completed"
        ? [{ role: "assistant" as const, content: last.request.content }] : [])] : [];
    // Bound context along with the session ledger, always beginning at a user turn.
    const recent = history.slice(-SESSION_LIMIT * 2);
    if (recent[0]?.role === "assistant") recent.shift();
    void generate([...recent, { role: "user", content: text.trim() }], model, text.trim());
  }
  const requests = messages.flatMap(m => m.role === "assistant" ? [m.request] : []);
  return {
    messages, requests, selected: requests.find(r => r.localId === selected), select,
    health, active, send,
    abort: () => {
      controller.current?.abort();
      const now = performance.now();
      setMessages(current => current.map(m => m.role === "assistant" && isRunning(m.request)
        ? { ...m, request: cancel(m.request, now) } : m));
    },
    retry: (r: RequestRecord) => { if (!active) void generate(r.history, r.model); },
    clear: () => { if (!controller.current) { setMessages([]); select(null); } },
  };
}
