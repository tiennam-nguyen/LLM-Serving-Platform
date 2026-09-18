"use client";
import { useRef, useState } from "react";
import type { RequestRecord } from "../../lib/telemetry/request-record";

export function MessageActions({ request, active, inspect, retry }: {
  request: RequestRecord; active: boolean; inspect: () => void; retry: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [notice, setNotice] = useState("");
  const trigger = useRef<HTMLButtonElement>(null);
  async function copy(value: string) {
    try { await navigator.clipboard.writeText(value); setNotice("Copied"); }
    catch { setNotice("Clipboard unavailable. Select and copy the text manually."); }
    setOpen(false); trigger.current?.focus();
  }
  return <div className="message-actions" onKeyDown={e => {
    if (e.key === "Escape") { setOpen(false); trigger.current?.focus(); }
  }}>
    <button ref={trigger} className="micro" aria-label={`Actions for request ${request.sequence}`}
      aria-expanded={open} onClick={() => setOpen(!open)}>⋯</button>
    {open && <div className="action-list recess" aria-label="Response actions">
      <button onClick={() => { inspect(); setOpen(false); trigger.current?.focus(); }}>Inspect request</button>
      <button disabled={!request.content} onClick={() => void copy(request.content)}>Copy response</button>
      <button disabled={!request.requestId} onClick={() => void copy(request.requestId!)}>Copy request ID</button>
      <button disabled={active} onClick={() => { retry(); setOpen(false); }}>Retry</button>
    </div>}
    <span className="copy-notice" role="status">{notice}</span>
  </div>;
}
