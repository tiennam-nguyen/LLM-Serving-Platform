import type { Health } from "../../lib/gateway/types";

export function TopBar({ model, health }: { model: string; health: Health }) {
  return <header className="topbar">
    <div className="brand"><span className="brand-mark" aria-hidden="true">▥</span><div>
      <h1>LLM SERVING PLATFORM</h1><p className="etch">HW-REV / 03 <span>DATA PLANE</span></p>
    </div></div>
    <div className="model-instrument recess"><span className="label">CONFIGURED MODEL</span>
      <strong title={model}>{model || "NOT CONFIGURED"}</strong></div>
    <div className="health" role="status"><span className={`led ${health.toLowerCase()}`} aria-hidden="true" />
      <span>GATEWAY / WORKER <b>{health}</b></span></div>
  </header>;
}
