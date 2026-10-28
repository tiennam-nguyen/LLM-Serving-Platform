import { milliseconds, type RequestRecord } from "../../lib/telemetry/request-record";
import { Plate } from "../shell/AppFrame";

function RequestLifecycle({ request: r }: { request: RequestRecord }) {
  const end = r.status === "cancelled" ? "ABORTED" : r.status === "failed" ? "FAILED" : "COMPLETED";
  const steps: [string, number | undefined][] = [
    ["SUBMITTED", r.submittedAt], ["GATEWAY ACCEPTED", r.acceptedAt],
    ["FIRST CONTENT", r.firstContentAt], [end, r.completedAt],
  ];
  return <ol className="lifecycle">{steps.map(([label, time]) => <li key={label} className={time === undefined ? "unobserved" : "observed"}>
    <span>{label}</span><b>{time === undefined ? "—" : `+${milliseconds(r.submittedAt, time)}`}</b>
  </li>)}</ol>;
}

export function RequestInspector({ request, requests, select }: {
  request?: RequestRecord; requests: RequestRecord[]; select: (id: string) => void;
}) {
  const workerId = request?.workerId ?? request?.serverTrace?.worker_id ?? "NOT REPORTED";
  const st = request?.serverTrace;
  return <aside aria-label="Request inspector"><Plate className="inspector">
    <header className="panel-header"><h2>REQUEST INSTRUMENTATION <span>{"// SPEC TRACE"}</span></h2>
      <span className={`status-pill ${request?.status ?? "idle"}`}>{request?.status === "cancelled" ? "ABORTED" : request?.status.toUpperCase() ?? "IDLE"}</span></header>
    <div className="inspector-body">
      <p className="label etch">SELECTED REQUEST / {request ? String(request.sequence).padStart(2, "0") : "—"}</p>
      <div className="metrics">
        <div className="gauge recess hot"><span className="label">FIRST CONTENT</span><strong>{milliseconds(request?.submittedAt, request?.firstContentAt)}</strong><small>CLIENT OBSERVED</small></div>
        <div className="gauge recess"><span className="label">E2E DURATION</span><strong>{milliseconds(request?.submittedAt, request?.completedAt)}</strong><small>{request?.completedAt !== undefined ? "UNTIL TERMINAL STATE" : "AWAITING TERMINAL STATE"}</small></div>
      </div>
      <section className="instrument-section"><h3>CLIENT-OBSERVED LIFECYCLE</h3>
        {request ? <RequestLifecycle request={request} /> : <p className="unavailable">Select an assistant response or session row.</p>}
      </section>
      <dl className="trace recess-dark">
        <div><dt>REQUEST ID</dt><dd className="signal">{request?.requestId ?? "NOT REPORTED"}</dd></div>
        <div><dt>WORKER ID</dt><dd className="signal">{workerId}</dd></div>
        <div><dt>MODEL</dt><dd>{request?.model ?? "—"}</dd></div>
        <div className="usage-grid">{([
          ["PROMPT TOKENS", request?.usage?.promptTokens], ["COMPLETION TOKENS", request?.usage?.completionTokens], ["TOTAL TOKENS", request?.usage?.totalTokens],
        ] as const).map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value ?? "—"}</dd></div>)}</div>
        <div className="trace-note">
          Usage: {request?.usage ? "REPORTED FIELDS ONLY" : "NOT REPORTED"}<br />
          {st ? `Server TTFT: ${st.metrics_ms.ttft_ms ?? "—"} ms · E2E: ${st.metrics_ms.e2e_ms ?? "—"} ms · Routing: ${st.metrics_ms.routing_ms ?? "—"} ms` : "Hardware / server phases: NOT REPORTED"}
        </div>
      </dl>
      <section className="instrument-section"><h3>SESSION REQUESTS <span>/ LAST 50</span></h3>
        <table className="ledger"><caption className="sr-only">Requests initiated in this browser session</caption>
          <thead><tr><th scope="col">SEQ</th><th scope="col">STATUS</th><th scope="col">FIRST</th><th scope="col">E2E</th></tr></thead>
          <tbody>{[...requests].reverse().map(r => <tr key={r.localId} onClick={() => select(r.localId)} className={request?.localId === r.localId ? "selected" : ""}>
            <td><button aria-label={`Inspect request ${r.sequence}`} aria-pressed={request?.localId === r.localId} onClick={() => select(r.localId)}>{String(r.sequence).padStart(2, "0")}</button></td>
            <td>{r.status === "cancelled" ? "ABORTED" : r.status.toUpperCase()}</td>
            <td>{milliseconds(r.submittedAt, r.firstContentAt)}</td><td>{milliseconds(r.submittedAt, r.completedAt)}</td>
          </tr>)}</tbody>
        </table>
        {!requests.length && <p className="unavailable">NO REQUESTS IN THIS SESSION</p>}
      </section>
      <p className="deck-foot etch">LOCAL SESSION ONLY · SERVER TRACE WHEN AVAILABLE</p>
    </div>
  </Plate></aside>;
}
