# Instrumentation Deck / Week 3 Playground

Next.js, React, TypeScript, and CSS operator console for the **Adaptive LLM Serving Platform**.

## Run Locally

Prerequisites: Node.js compatible with the installed Next.js package, npm, and a running FastAPI Gateway (`http://127.0.0.1:8080`).

From `apps/web`:

```powershell
npm ci
Copy-Item .env.example .env.local
# Edit .env.local with the exact model served by your worker.
npm run dev
```

Open `http://127.0.0.1:3000`. Missing model configuration disables Send. A configured model is not a claim that it is online; the health LED polls the gateway and current worker.

### Environment Configuration

The **canonical Week 3 communication path is direct Browser → FastAPI Gateway** (`http://127.0.0.1:8080`) using the backend `CORSMiddleware` (which exposes `X-Request-ID` and `X-Worker-ID`).

| Variable | Meaning |
| --- | --- |
| `NEXT_PUBLIC_DEFAULT_MODEL` | Required to send. Exact served model name (e.g., `Qwen/Qwen2.5-0.5B-Instruct`); no invented fallback. |
| `NEXT_PUBLIC_GATEWAY_URL` | **Canonical direct browser-to-gateway origin.** Defaults to `http://127.0.0.1:8080` in `src/lib/gateway/client.ts` when omitted. Set explicitly to `/api/gateway` only if you want to route through the optional Next.js server relay. |
| `GATEWAY_URL` | Server-only upstream gateway origin used when the optional Next.js `/api/gateway/*` relay routes (`/health`, `/v1/chat/completions`, `/v1/traces/[requestId]`) are invoked. Defaults to `http://127.0.0.1:8080`. |

Restart development after environment changes. `NEXT_PUBLIC_*` values are embedded at build time; set them before production build. Never put credentials in these values.

```powershell
npm run lint
npm run typecheck
npm test
npm run build
npm start
```

## Browser Checks

```powershell
npx playwright install chromium
npm run test:browser
```

Stop any `next dev` instance in this directory first. The suite starts Next on port `3001` and a **test-only wire fixture** on `8081`. It tests progressive SSE, upstream cancellation, multi-turn history, request ID capture, worker ID display, server trace correlation, inspector selection, errors/retry, and phone layout.

To verify direct-browser CORS against a running real FastAPI gateway:

```powershell
$env:CORS_PROBE_URL='http://127.0.0.1:8080/health'
npm run test:browser -- tests/browser/origin.spec.ts
Remove-Item Env:CORS_PROBE_URL
```

This test verifies that direct browser fetch succeeds with HTTP 200 and exposes `X-Worker-ID` via FastAPI `CORSMiddleware`.

## Behavior and Architecture

- `useChatSession` owns multi-turn branches, cancellation, health polling, and per-generation records. Retry creates a new assistant generation from the selected request's original history/model and retains the earlier response. Failed/aborted assistant text remains visible in the UI but is excluded from future conversation context.
- `gateway/sse.ts` handles arbitrary chunk boundaries, CRLF, multiline data, JSON deltas, explicit usage, errors, and `[DONE]`.
- `RequestInspector.tsx` displays **client-observed data** (`FIRST CONTENT`, `E2E DURATION`, lifecycle milestones) alongside **server trace telemetry** (`X-Request-ID`, `X-Worker-ID`, server `ttft_ms`, `e2e_ms`, `routing_ms`) fetched from `GET /v1/traces/{request_id}`.
- Token usage only displays valid nonnegative integer fields explicitly reported by the stream; missing usage or unverified token counts display `—`.
