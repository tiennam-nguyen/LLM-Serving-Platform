import { expect, test } from "@playwright/test";

test("direct browser request succeeds with Backend CORS and exposes X-Worker-ID", async ({ page }) => {
  await page.goto("/");
  const url = process.env.CORS_PROBE_URL || "http://127.0.0.1:8081/health";
  const result = await page.evaluate(async (targetUrl) => {
    const resp = await fetch(targetUrl);
    return {
      status: resp.status,
      workerId: resp.headers.get("X-Worker-ID"),
      body: await resp.json(),
    };
  }, url);
  expect(result.status).toBe(200);
  expect(result.body.status).toBe("healthy");
  expect(result.workerId).toBeTruthy();
});

test("workers topology page renders control-plane registry and navigates back to playground", async ({ page }) => {
  await page.goto("/workers");
  await expect(page.getByRole("heading", { name: "WORKERS / CONTROL PLANE" })).toBeVisible();
  await expect(page.getByText("fixture-worker-0", { exact: true })).toBeVisible();
  await expect(page.getByText("fixture-worker-1", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Refresh" })).toBeEnabled();
  await page.getByRole("link", { name: /Playground/ }).click();
  await expect(page).toHaveURL(/\/$/);
});
