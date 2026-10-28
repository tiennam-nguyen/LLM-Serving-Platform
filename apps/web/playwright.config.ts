import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests/browser", workers: 1, use: { baseURL: "http://127.0.0.1:3001", headless: true },
  webServer: [
    { command: "node tests/fixture.mjs", url: "http://127.0.0.1:8081/health", reuseExistingServer: false },
    {
      command: "npm run dev -- --port 3001",
      url: "http://127.0.0.1:3001",
      reuseExistingServer: false,
      env: {
        NEXT_PUBLIC_DEFAULT_MODEL: "fixture-model",
        NEXT_PUBLIC_GATEWAY_URL: "http://127.0.0.1:8081",
        GATEWAY_URL: "http://127.0.0.1:8081",
      },
      timeout: 120000,
    },
  ],
});
