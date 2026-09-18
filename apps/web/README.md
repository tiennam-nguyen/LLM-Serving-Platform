# Instrumentation deck / Week 3 playground

Next.js, React, TypeScript and ordinary CSS. The metal visual system is adapted from the approved `instrumentation-deck.html`; its simulations and fake metrics are not used.

## Run locally

Prerequisites: Node.js compatible with the installed Next.js package (`node_modules/next/package.json:engines`), npm, and a reachable gateway. Verified frontend toolchain: Node 24.13.0 / npm 11.6.2. Gateway runtime prerequisites remain in the repository runbooks.

From `apps/web`:

```powershell
npm install
Copy-Item .env.example .env.local
# Edit .env.local with the exact model served by your worker.
npm run dev
```

Open http://127.0.0.1:3000. Missing model configuration disables Send. A configured model is not a claim that it is online; the health LED checks the gateway/current worker.

| Variable | Meaning |
| --- | --- |
| `NEXT_PUBLIC_DEFAULT_MODEL` | Required to send. Exact served model name; no invented fallback or model discovery. |
| `GATEWAY_URL` | Server-only gateway origin. Defaults to `http://127.0.0.1:8080`. Relay forwards only `/health` and `/v1/chat/completions`. |
| `NEXT_PUBLIC_GATEWAY_URL` | Optional direct browser gateway origin. Omit to use the same-origin relay. Direct mode needs backend CORS. |

Restart development after environment changes. `NEXT_PUBLIC_*` values are embedded at build time; set them before production build. Never put credentials in these values. Default dev/start commands bind to loopback. No authentication or public deployment is included.

```powershell
npm run lint
npm run typecheck
npm test
npm run build
npm start
```

Use `npm ci` for reproducible installation from the lockfile.

## Browser checks

```powershell
npx playwright install chromium
npm run test:browser
```

Stop any `next dev` instance in this directory first. The suite starts Next on port 3001 and a **test-only wire fixture** on 8081. It tests progressive SSE, upstream cancellation, multi-turn history, request ID capture, inspector selection, errors/retry, and phone layout. It does not prove live vLLM/GPU inference. Screenshots go to ignored `test-results/`.

To reproduce direct-browser CORS against a running real gateway:

```powershell
$env:CORS_PROBE_URL='http://127.0.0.1:8080/health'
npm run test:browser -- tests/browser/origin.spec.ts
Remove-Item Env:CORS_PROBE_URL
```

This negative test expects missing-CORS failure. If Person 2 enables direct browser access, update that expectation. Direct mode needs an explicit permitted frontend origin, GET/POST/OPTIONS support, Content-Type/Accept request headers, and `Access-Control-Expose-Headers: X-Request-ID`. No backend CORS settings are changed here.

The relay was verified with delayed fixture events: first content reached the browser before completion; Abort closed the upstream socket. It forwards bytes without collecting the stream into text and preserves `X-Request-ID`. Before-header failures become HTTP 502; after-header failures retain partial output. It has a 10-second response-header deadline and no synthetic generation deadline. Hosting proxy buffering/timeouts need separate verification. Cross-origin POSTs are rejected against the browser Host header and request scheme; TLS-terminating deployments must preserve the external scheme or adapt and verify this check.

## Behavior and architecture

`useChatSession` owns branches, cancellation and per-generation records. Retry creates a new assistant generation from the selected request's original history/model and retains the earlier response. Subsequent prompts continue the latest generation's branch. Failed/aborted assistant text remains visible but is excluded from future context. Clear Chat resets the feed/ledger. The latest 50 requests and at most 100 prior context messages per submission are retained in memory; refreshing clears the session. No persistence or global history exists.

`gateway/sse.ts` handles chunks, CRLF, multiline data, JSON deltas, explicit usage, errors and `[DONE]`. Malformed records show a warning without crashing the feed. EOF without `[DONE]` is a failure. Markdown supports prose, line breaks, inline code and fenced code; raw HTML is not enabled.

The right instrumentation deck displays **client-observed data for the selected request**. It is not the future system-wide observability dashboard. Timing uses `performance.now()`: submission, successful headers, first nonempty assistant delta, and terminal completion/failure/abort. “Gateway accepted” means successful response headers, not worker acceptance. FIRST CONTENT is not server TTFT; E2E includes the browser/relay path. An unfinished duration displays `—`.

Request ID comes from response headers; model comes from the submitted configuration. Usage only displays valid nonnegative integer fields reported by the stream. Missing usage, worker identity, GPU hardware, queue/prefill/decode durations and rates remain unavailable. No token estimation is performed. Health runs initially, after failures, and every 30 seconds, with an 8-second timeout.

See [IMPLEMENTATION.md](IMPLEMENTATION.md) for observed verification and live-environment limits.
