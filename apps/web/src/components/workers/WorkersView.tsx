"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { fetchWorkers, type WorkerRecord } from "../../lib/control/client";
import { Plate } from "../shell/AppFrame";
import "./workers.css";

export function WorkersView() {
  const [records, setRecords] = useState<WorkerRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [refresh, setRefresh] = useState(0);
  useEffect(() => {
    let disposed = false;
    const abort = new AbortController();
    const timeout = setTimeout(() => abort.abort(), 10000);
    void fetchWorkers(abort.signal).then(value => {
      if (!abort.signal.aborted) setRecords(value);
    }).catch(reason => {
      if (!disposed) setError(abort.signal.aborted ? "Workers request timed out. Try Refresh."
        : reason instanceof Error ? reason.message : "Workers request failed");
    }).finally(() => { clearTimeout(timeout); if (!disposed) setLoading(false); });
    return () => { disposed = true; clearTimeout(timeout); abort.abort(); };
  }, [refresh]);
  return <main className="workers-view">
    <nav aria-label="Views"><Link href="/">← Playground</Link></nav>
    <Plate><header className="panel-header"><h1>WORKERS / CONTROL PLANE</h1>
      <button disabled={loading} onClick={() => { setLoading(true); setError(""); setRefresh(n => n + 1); }}>Refresh</button>
    </header><div className="workers-body">
      {loading ? <p role="status">Loading workers…</p> : error ? <p className="error-frame" role="alert">{error}</p>
        : !records.length ? <p role="status">No workers registered.</p> : <ul className="worker-list">
          {records.map(record => <li className="recess" key={record.worker_id}><h2>{record.worker_id}</h2>
            <dl><dt>MODEL</dt><dd>{record.model}</dd><dt>ENDPOINT</dt><dd>{record.endpoint}</dd>
              <dt>HEALTH STATUS</dt><dd className={`worker-status ${record.status.toLowerCase()}`}>{record.status}</dd>
              <dt>LAST CHECKED (UTC)</dt><dd>{record.last_checked_at
                ? <time dateTime={record.last_checked_at}>{new Date(record.last_checked_at).toISOString()}</time> : "Not checked yet"}</dd>
            </dl></li>)}
        </ul>}
    </div></Plate>
  </main>;
}
