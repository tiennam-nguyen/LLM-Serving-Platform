import { expect, test } from "@playwright/test";

test("direct browser request is blocked by missing CORS headers", async ({ page }) => {
  const errors: string[] = [];
  page.on("console", msg => { if (msg.type() === "error") errors.push(msg.text()); });
  await page.goto("/");
  const url = process.env.CORS_PROBE_URL || "http://127.0.0.1:8081/health";
  const result = await page.evaluate(async (url) => {
    try { await fetch(url); return "unexpected success"; }
    catch (error) { return String(error); }
  }, url);
  expect(result).toContain("Failed to fetch");
  console.log("DIRECT BROWSER RESULT:", url, result, errors.filter(e => e.includes("CORS")));
  expect(errors.some(e => e.includes("CORS"))).toBe(true);
});
