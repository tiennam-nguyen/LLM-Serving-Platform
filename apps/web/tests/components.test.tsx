import { useState } from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, it, vi } from "vitest";
import { Composer } from "../src/components/chat/Composer";
import { RequestInspector } from "../src/components/telemetry/RequestInspector";
import type { RequestRecord } from "../src/lib/telemetry/request-record";

it("blocks empty/whitespace, handles Enter and Shift+Enter, and disables unconfigured send", async () => {
  const user = userEvent.setup(); const send = vi.fn();
  const { rerender } = render(<Composer active={false} configured onSend={send} onAbort={vi.fn()} />);
  const prompt = screen.getByRole("textbox");
  expect(screen.getByRole("button", { name: /SEND/ })).toBeDisabled();
  await user.type(prompt, "   {Enter}"); expect(send).not.toHaveBeenCalled();
  expect(screen.getByRole("button", { name: /SEND/ })).toBeDisabled();
  await user.clear(prompt); await user.type(prompt, "hello{Shift>}{Enter}{/Shift}world");
  expect(prompt).toHaveValue("hello\nworld"); expect(send).not.toHaveBeenCalled();
  await user.keyboard("{Enter}"); expect(send).toHaveBeenCalledWith("hello\nworld");
  expect(prompt).toHaveValue("");
  rerender(<Composer active={false} configured={false} onSend={send} onAbort={vi.fn()} />);
  await user.type(prompt, "valid"); expect(screen.getByRole("button", { name: /SEND/ })).toBeDisabled();
});
it("uses a real abort button during generation", async () => {
  const abort = vi.fn(); render(<Composer active configured onSend={vi.fn()} onAbort={abort} />);
  await userEvent.click(screen.getByRole("button", { name: /ABORT/ })); expect(abort).toHaveBeenCalledOnce();
});
it("selecting a ledger request switches all inspector details", async () => {
  const requests: RequestRecord[] = [1, 2].map(sequence => ({
    sequence, localId: String(sequence), requestId: `gateway-${sequence}`, model: `model-${sequence}`,
    status: "completed", submittedAt: 0, completedAt: sequence * 100, content: "", history: [], warnings: [],
  }));
  function Harness() {
    const [id, select] = useState("1");
    return <RequestInspector requests={requests} request={requests.find(r => r.localId === id)} select={select} />;
  }
  render(<Harness />); expect(screen.getByText("gateway-1")).toBeVisible();
  await userEvent.click(screen.getByRole("button", { name: "Inspect request 2" }));
  expect(screen.queryByText("gateway-1")).not.toBeInTheDocument();
  expect(screen.getByText("gateway-2")).toBeVisible(); expect(screen.getByText("model-2")).toBeVisible();
  expect(screen.getByText(/Usage: NOT REPORTED/)).toBeVisible();
});
