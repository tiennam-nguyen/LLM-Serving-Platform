import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { fetchWorkers, parseWorkers } from "../src/lib/control/client";
import { WorkersView } from "../src/components/workers/WorkersView";

const row = { worker_id: "mock-a", endpoint: "mock://a", model: "model", status: "STARTING", last_checked_at: null };
afterEach(() => { vi.unstubAllGlobals(); vi.unstubAllEnvs(); });
it("validates records, status, duplicates and aware timestamps", () => {
  expect(parseWorkers([row])).toEqual([row]);
  for (const value of [{ workers: [] }, [null], [{ ...row, status: "OFFLINE" }], [{ ...row, model: "" }],
    [{ ...row, last_checked_at: "2026-11-02T01:00:00" }], [{ ...row, last_checked_at: "bad" }], [row, row]]) {
    expect(() => parseWorkers(value)).toThrow();
  }
});
it("uses configured control URL, falls back to gateway, propagates HTTP errors", async () => {
  const fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => [row] });
  vi.stubGlobal("fetch", fetch);
  vi.stubEnv("NEXT_PUBLIC_CONTROL_API_URL", "http://control/");
  await fetchWorkers(); expect(fetch.mock.calls[0][0]).toBe("http://control/control/v1/workers");
  vi.stubEnv("NEXT_PUBLIC_CONTROL_API_URL", ""); vi.stubEnv("NEXT_PUBLIC_GATEWAY_URL", "http://gateway/");
  await fetchWorkers(); expect(fetch.mock.calls[1][0]).toBe("http://gateway/control/v1/workers");
  vi.stubEnv("NEXT_PUBLIC_GATEWAY_URL", "");
  await fetchWorkers(); expect(fetch.mock.calls[2][0]).toBe("http://127.0.0.1:8080/control/v1/workers");
  fetch.mockResolvedValue({ ok: false, status: 404 });
  await expect(fetchWorkers()).rejects.toThrow("HTTP 404");
});
it("shows loading then records with all three health states, and refreshes", async () => {
  let resolve!: (value: unknown) => void;
  const fetch = vi.fn().mockReturnValueOnce(new Promise(r => { resolve = r; }))
    .mockResolvedValue({ ok: true, json: async () => [] });
  vi.stubGlobal("fetch", fetch);
  render(<WorkersView />);
  expect(screen.getByRole("status")).toHaveTextContent("Loading");
  expect(screen.getByRole("button", { name: "Refresh" })).toBeDisabled();
  resolve({ ok: true, json: async () => [row, { ...row, worker_id: "mock-b", status: "UNHEALTHY", last_checked_at: "2026-11-02T00:00:00Z" }, { ...row, worker_id: "mock-c", status: "HEALTHY" }] });
  expect(await screen.findByText("UNHEALTHY")).toBeVisible();
  expect(screen.getByText("STARTING")).toBeVisible(); expect(screen.getByText("HEALTHY")).toBeVisible();
  expect(screen.getAllByText("mock://a")).toHaveLength(3);
  expect(screen.getByText("2026-11-02T00:00:00.000Z")).toBeVisible();
  await userEvent.click(screen.getByRole("button", { name: "Refresh" }));
  expect(await screen.findByText("No workers registered.")).toBeVisible();
  expect(fetch).toHaveBeenCalledTimes(2);
});
it("shows error without breaking navigation and recovers on refresh", async () => {
  const fetch = vi.fn().mockRejectedValueOnce(new Error("network unavailable"))
    .mockResolvedValue({ ok: true, json: async () => [row] });
  vi.stubGlobal("fetch", fetch);
  render(<WorkersView />);
  expect(await screen.findByRole("alert")).toHaveTextContent("network unavailable");
  expect(screen.getByRole("link", { name: /Playground/ })).toHaveAttribute("href", "/");
  await userEvent.click(screen.getByRole("button", { name: "Refresh" }));
  expect(await screen.findByText("mock-a")).toBeVisible();
  await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument());
});
