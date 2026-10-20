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
