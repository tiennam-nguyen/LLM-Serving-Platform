export const workerStatuses = ["STARTING", "HEALTHY", "UNHEALTHY"] as const;
export type WorkerStatus = typeof workerStatuses[number];
export type WorkerRecord = {
  worker_id: string; endpoint: string; model: string;
  status: WorkerStatus; last_checked_at: string | null;
};

export function parseWorkers(payload: unknown): WorkerRecord[] {
  if (!Array.isArray(payload)) throw new Error("Invalid worker list response");
  const ids = new Set<string>();
  return payload.map(value => {
    if (!value || typeof value !== "object") throw new Error("Invalid worker record");
    const record = value as Record<string, unknown>;
    for (const key of ["worker_id", "endpoint", "model"]) {
      if (typeof record[key] !== "string" || !record[key].trim()) throw new Error(`Invalid worker ${key}`);
    }
    if (!workerStatuses.includes(record.status as WorkerStatus)) throw new Error("Invalid worker status");
    const checked = record.last_checked_at;
    if (checked !== null && (typeof checked !== "string" || !/(Z|[+-]\d{2}:\d{2})$/.test(checked) || !Number.isFinite(Date.parse(checked)))) {
      throw new Error("Invalid worker last checked time");
    }
    const id = record.worker_id as string;
    if (ids.has(id)) throw new Error("Duplicate worker identity in response");
    ids.add(id);
    return { worker_id: id, endpoint: record.endpoint as string, model: record.model as string,
      status: record.status as WorkerStatus, last_checked_at: checked as string | null };
  });
}

export async function fetchWorkers(signal?: AbortSignal): Promise<WorkerRecord[]> {
  const base = (process.env.NEXT_PUBLIC_CONTROL_API_URL?.trim()
    || process.env.NEXT_PUBLIC_GATEWAY_URL?.trim() || "http://127.0.0.1:8080").replace(/\/$/, "");
  const response = await fetch(`${base}/control/v1/workers`, { signal, cache: "no-store" });
  if (!response.ok) throw new Error(`Workers request failed (HTTP ${response.status})`);
  return parseWorkers(await response.json());
}
