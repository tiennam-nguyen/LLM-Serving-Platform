"use client";
import { useLayoutEffect, useRef, useState } from "react";

export function Composer({ active, configured, onSend, onAbort }: {
  active: boolean; configured: boolean; onSend: (text: string) => void; onAbort: () => void;
}) {
  const [text, setText] = useState("");
  const textarea = useRef<HTMLTextAreaElement>(null);
  useLayoutEffect(() => {
    const field = textarea.current;
    if (field) { field.style.height = "auto"; field.style.height = `${Math.min(field.scrollHeight, 180)}px`; }
  }, [text]);
  function submit() {
    if (!active && configured && text.trim()) { onSend(text); setText(""); }
  }
  return <form className="dock" onSubmit={e => { e.preventDefault(); submit(); }}>
    <label className="label" htmlFor="prompt">INPUT CHANNEL / PROMPT</label>
    <div className="composer recess">
      <textarea id="prompt" ref={textarea} rows={2} value={text}
        placeholder={configured ? "Send an instruction to the configured model…" : "Set NEXT_PUBLIC_DEFAULT_MODEL to begin."}
        onChange={e => setText(e.target.value)}
        onKeyDown={e => {
          if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); submit(); }
        }} />
      {active ? <button className="action abort" type="button" onClick={onAbort}>■ ABORT</button>
        : <button className="action" type="submit" disabled={!configured || !text.trim()}>↑ SEND</button>}
    </div>
    <div className="dock-note etch"><span>ENTER / SEND · SHIFT + ENTER / NEWLINE</span><span>STREAM / SSE</span></div>
  </form>;
}
