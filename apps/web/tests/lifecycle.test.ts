import { expect, it } from "vitest";
import { cancel, milliseconds, transition, type RequestRecord } from "../src/lib/telemetry/request-record";

export const record = (localId = "local", sequence = 1): RequestRecord => ({
  localId, sequence, requestId: null, model: "test-model", status: "submitting",
  submittedAt: 0, content: "", history: [], warnings: [],
});
it("submitting -> streaming -> completed and first-content recorded once", () => {
  let r = transition(record(), { type: "content", text: "One" }, 10);
  expect(r.status).toBe("streaming");
  r = transition(r, { type: "content", text: " two" }, 20);
  expect(r.firstContentAt).toBe(10);
  expect(r.content).toBe("One two");
  r = transition(r, { type: "done" }, 30);
  expect(r.status).toBe("completed");
  expect(milliseconds(r.submittedAt, r.completedAt)).toBe("30 ms");
});
it("submitting -> failed stays failed after DONE", () => {
  const r = transition(record(), { type: "error", message: "down", code: 503 }, 10);
  expect(transition(r, { type: "done" }, 20)).toEqual(r);
  expect(r.status).toBe("failed");
  expect(r.firstContentAt).toBeUndefined();
});
it("streaming -> cancelled retains partial output and rejects late events", () => {
  const r = cancel(transition(record(), { type: "content", text: "partial" }, 5), 9);
  expect(r.status).toBe("cancelled");
  expect(r.content).toBe("partial");
  expect(transition(r, { type: "content", text: "late" }, 12)).toEqual(r);
});
it("empty delta never marks first content and missing timing stays unavailable", () => {
  expect(transition(record(), { type: "content", text: "" }, 0)).toEqual(record());
  expect(milliseconds(0, undefined)).toBe("—");
  expect(milliseconds(0, 0)).toBe("0 ms");
});
