# Week 8 Open-Loop Pilot — Dual-T4 Load-Level Selection Evidence

> **Artifact status**: Member A pilot-only load-selection report (`pilot_metrics.json`, `pilot_report.json`, `pilot_summary.csv`, `combined_pilot_selection.json`, `combined_pilot_summary.csv`). This does **not** replace Member B's final `summary.json`, 3-profile workloads, plotting scripts, or 54-cell benchmark campaign.

---

## 1. Historical Week 7 Context (Not W8 Controlled Evidence)

> **Scope label**: *Historical W7 closed/concurrency-limited context (`concurrency=2`, `max_tokens=64`, `temperature=0`, `N=20`, `warmup=2`, commit `df03a27`) — NOT W8 controlled open-loop evidence.*

Recomputed directly from raw traces under `experiments/raw/week7/live/` using `compute_percentile` (NIST R-7):

| Policy | Run ID | Completed | Span (s) | Achieved req/s | E2E p50 / p95 / p99 (ms) | TTFT p50 / p95 / p99 (ms) | Routing p50 / p95 / p99 (ms) | Worker Distribution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `round_robin` | `20261122T045731Z-w7-t4-round-robin-df03a27-14295e` | 20/20 | 17.464 | 1.145 | 1774.005 / 1988.837 / 2065.543 | 49.980 / 90.067 / 90.621 | 0.041 / 0.058 / 0.060 | `worker-vllm-0: 10`, `worker-vllm-1: 10` |
| `least_loaded` | `20261122T045755Z-w7-t4-least-loaded-df03a27-4ed5ce` | 20/20 | 21.243 | 0.941 | 2182.375 / 2532.726 / 2551.057 | 47.350 / 66.462 / 67.556 | 0.042 / 0.070 / 0.078 | `worker-vllm-0: 9`, `worker-vllm-1: 11` |

Because W7 held `runner.concurrency = 2` (1 active request per GPU replica, no continuous batching concurrency), its achieved throughput (`0.94–1.15 req/s`) established a conservative starting floor of **`1.0 RPS`** for the W8 open-loop sweep.

---

## 2. Pre-Declared Policy-Neutral Selection Criteria

Every candidate offered rate `R` is evaluated identically on paired `(round_robin, least_loaded)` runs sharing the exact same ordered steady workload (`workload_sha256`), monotonic schedule (`schedule_sha256`), model, vLLM configuration, generation settings, and warm-up procedure (`warmup.request_count = 2`).

1. **Run & Pair Validity**:
   - Both RR and LL runs finish with `run_status == "completed"`, zero `generator_limited`, `missing_trace == 0`, `dispatch_lag_max_ms <= 250.0`, identical `workload_sha256` and `schedule_sha256`, and `active_requests == 0` on all workers post-drain.
2. **Stability Criteria (Both Policies Must Satisfy All)**:
   - `noncompletion_rate <= 0.02`
   - Normalized drain throughput tracking: accounting for finite measurement window tail drain (`expected_no_queue_drain_rps = R * T / (T + baseline_e2e_p95_s)`), `completed_drain_rps >= 0.85 * expected_no_queue_drain_rps`.
   - Window-end backlog fraction: `backlog_at_window_end / N <= max(0.35, 1.5 * baseline_e2e_p95_s / T)` (prevents confusing Little's Law in-flight work `lambda * W` with unbounded queue growth).
   - E2E tail growth: `e2e_p95 <= 2.5 * baseline_e2e_p95` (relative to the `1.0 RPS` baseline `2000.885 ms`, threshold `5002.2 ms`).
   - TTFT tail growth: `ttft_p95 <= max(5.0 * baseline_ttft_p95, 400.0 ms)` (relative to `1.0 RPS` baseline `48.358 ms`, threshold `400.0 ms`).
3. **Common Load-Point Selection**:
   - **`low`**: Lowest common valid stable rate meeting `completed_drain_rps >= 0.92 * expected_no_queue_drain_rps` and `backlog_at_window_end / N <= 0.15`.
   - **`near_saturation`**: Highest common valid stable rate immediately preceding the first degraded/saturated transition.
   - **`medium`**: Tested common valid stable rate strictly between `low` and `near_saturation` closest to the geometric midpoint `sqrt(low * near_saturation)` in log space.

### Discovery from Pilot Run #1 (`1.0–20.0 RPS`)

Run #1 initially used an unnormalized `completed_drain_rps >= 0.90 * offered_rps` rule, which tripped at `17.0 RPS` (`15.30 / 17.00 = 0.900`). Inspection showed `20.0 RPS` still achieved `400/400` completions (`noncompletion = 0`), `TTFT p50 = 124–125 ms`, `E2E p50 = 2.90–2.91 s`, and `backlog_at_window_end = 60` (matching Little's Law `L = 20 * 2.90 = 58`). Because `N / (T + 2.35 s)` geometrically caps `completed_drain_rps` at `0.895 * offered_rps` on a `20 s` window, we updated `pilot_metrics.py` (commit `2621c80`) to normalize by `T / (T + baseline_e2e_p95_s)` and ran Pilot Run #2 across `20.0–44.0 RPS` (`window_ms = 10000` so `N <= 440 < 500`, preventing `InMemoryTraceStore` eviction).

---

## 3. Kaggle Dual-T4 Environment & Provenance

| Attribute | Pilot Run #1 (`runset_v1_1to20rps`) | Pilot Run #2 (`runset_v2_20to48rps`) |
| --- | --- | --- |
| **Kaggle kernel slug** | `tiennamnguyen11/llm-serving-week-8-open-loop-pilot` (v1) | `tiennamnguyen11/llm-serving-week-8-open-loop-pilot` (v2) |
| **Tested Git commit** | `b9c02ba99ea2219f03e67bd350de48b0c94ed94b` (`git_dirty: false`) | `2621c808ec879ca31f0dda7e64883a2f464ec1ce` (`git_dirty: false`) |
| **Bundle SHA-256** | `2de4abe4c06a07af78ddf58da5481b66625e362d141b445e60a9805c22dd28ce` | `692dff5e03b6ac9ffd6a5228628de91251d43d74526d33a720487d2b9159c244` |
| **Packaged script SHA-256** | `102c7ec653161d6f48a19d0ff2543d757b0b14534be84be46efddf1924708227` | `25a0722b93193d30d9fb197cbd10127c77f9fd49e8b8049d5e7c94d2788fb2c0` |
| **GPUs & Driver** | 2 × NVIDIA Tesla T4 (`15360 MiB` each), driver `580.178.04` | 2 × NVIDIA Tesla T4 (`15360 MiB` each), driver `580.178.04` |
| **Worker ↔ GPU bindings** | GPU 0 (`GPU-2f9a2908-...`) → `worker-vllm-0` (`:8000`)<br>GPU 1 (`GPU-f7d2ce3c-...`) → `worker-vllm-1` (`:8001`) | GPU 0 (`GPU-19a6d724-...`) → `worker-vllm-0` (`:8000`)<br>GPU 1 (`GPU-06cf16fc-...`) → `worker-vllm-1` (`:8001`) |
| **Software stack** | Python `3.12.16`, PyTorch `2.14.0+cu130`, CUDA `13.0`, vLLM `0.33.0`, Transformers `5.20.2` | Python `3.12.16`, PyTorch `2.14.0+cu130`, CUDA `13.0`, vLLM `0.33.0`, Transformers `5.20.2` |
| **Model & vLLM flags** | `Qwen/Qwen2.5-3B-Instruct`, `tp=1`, `float16`, `gpu_memory_utilization=0.85`, `max_model_len=4096` | `Qwen/Qwen2.5-3B-Instruct`, `tp=1`, `float16`, `gpu_memory_utilization=0.85`, `max_model_len=4096` |
| **Generation & Warm-up** | `temperature=0`, `max_tokens=64`, `top_p=1`, `warmup=2` | `temperature=0`, `max_tokens=64`, `top_p=1`, `warmup=2` |
| **Cache & Gateway control** | Both vLLM replicas restarted + fresh Gateway per policy run | Both vLLM replicas restarted + fresh Gateway per policy run |
| **Window & Runner guards** | `window=20000 ms`, `max_in_flight=180`, `max_dispatch_lag=250 ms`, `timeout=120 s` | `window=10000 ms`, `max_in_flight=350`, `max_dispatch_lag=250 ms`, `timeout=120 s` |
| **Candidate rates tested** | `1, 2, 4, 8, 12, 16, 17, 18, 20 RPS` (18 runs) | `20, 28, 32, 34, 36, 44 RPS` (12 runs; stopped at 44 RPS on clear saturation) |

### Local Oversized Service Logs (Excluded from Git, Retained Locally with SHA-256)

| File | Local Path (in worktree `.cache/`) | SHA-256 |
| --- | --- | --- |
| v1 `worker0.log` | `.cache/kaggle-week8/output-v1/week8/service-logs/worker0.log` | `E4FBAAB6CB2D4B1300DDACEDBE6CE49E9FBDD55C7B390AFE7F3BCDAAEEAB5046` |
| v1 `worker1.log` | `.cache/kaggle-week8/output-v1/week8/service-logs/worker1.log` | `350A323A78AB71E9C867D094B3A67B7D72E27812FB2F2E0E59D7C591444F89DE` |
| v2 `worker0.log` | `.cache/kaggle-week8/output-v2/week8/service-logs/worker0.log` | `1B0A518A1DD936DDD474EF2EB1BC44E493C7F707B3C774810305804E2325E217` |
| v2 `worker1.log` | `.cache/kaggle-week8/output-v2/week8/service-logs/worker1.log` | `C49CFC0BA292D6DF649ECEC070BB57923B4AB92CE1583298760CAF6AC7DAA887` |
| v2 Kaggle notebook log | `.cache/kaggle-week8/output-v2/llm-serving-week-8-open-loop-pilot.log` | `3A6DB9C7F0CCC7DC091E191607F37EF592D025C86A57EF3D8455FB3634D74A40` |

---

## 4. Pilot Candidate Table (All 30 Dual-T4 Runs)

> All P50/P95/P99 values use NIST R-7 linear interpolation from `traces.jsonl` (`routing` across all `N` observed traces; `TTFT`/`E2E` across all `N` completed traces). Sample counts `n = planned` (`20` at `1 RPS` up to `440` at `44 RPS`). Low-N P99 (especially `n = 20–80`) has limited statistical precision.

| Runset (Win) | Order | Policy | RPS | Sent/Plan | Comp | Drain RPS | TTFT p50/p95/p99 (ms) | E2E p50/p95/p99 (ms) | Routing p50/p95/p99 (ms) | Lag p95/p99/max (ms) | Noncomp | Backlog | Workers (w0/w1) | Valid |
| --- | --- | --- | ---: | --- | ---: | ---: | --- | --- | --- | --- | ---: | ---: | --- | --- |
| v1 (20s) | RR→LL | `round_robin` | 1.0 | 20/20 | 20 | 0.965 | 46.0 / 47.1 / 47.3 | 1727.1 / 1745.7 / 1748.3 | 0.044 / 0.059 / 0.061 | 1.60 / 1.79 / 1.83 | 0.0 | 1 | 10 / 10 | valid |
| v1 (20s) | RR→LL | `least_loaded` | 1.0 | 20/20 | 20 | 0.964 | 45.3 / 48.4 / 49.4 | 1765.6 / 2000.9 / 2098.5 | 0.044 / 0.074 / 0.092 | 2.22 / 2.28 / 2.29 | 0.0 | 2 | 10 / 10 | valid |
| v1 (20s) | LL→RR | `least_loaded` | 2.0 | 40/40 | 40 | 1.895 | 91.4 / 103.3 / 112.4 | 1900.9 / 1954.3 / 2084.5 | 0.045 / 0.063 / 0.081 | 1.11 / 1.14 / 1.14 | 0.0 | 3 | 20 / 20 | valid |
| v1 (20s) | LL→RR | `round_robin` | 2.0 | 40/40 | 40 | 1.882 | 91.4 / 102.9 / 113.2 | 1910.9 / 1966.2 / 2188.9 | 0.043 / 0.086 / 0.106 | 1.70 / 2.02 / 2.11 | 0.0 | 3 | 20 / 20 | valid |
| v1 (20s) | RR→LL | `round_robin` | 4.0 | 80/80 | 80 | 3.690 | 89.9 / 104.4 / 108.3 | 1972.8 / 2004.8 / 2036.7 | 0.044 / 0.093 / 0.683 | 1.28 / 2.12 / 2.22 | 0.0 | 8 | 40 / 40 | valid |
| v1 (20s) | RR→LL | `least_loaded` | 4.0 | 80/80 | 80 | 3.661 | 89.5 / 104.4 / 107.2 | 1980.4 / 2011.0 / 2064.0 | 0.043 / 0.068 / 0.205 | 1.79 / 2.14 / 2.15 | 0.0 | 7 | 40 / 40 | valid |
| v1 (20s) | LL→RR | `least_loaded` | 8.0 | 160/160 | 160 | 7.302 | 95.2 / 110.5 / 113.0 | 2129.1 / 2218.6 / 2246.8 | 0.044 / 0.087 / 0.111 | 1.89 / 3.27 / 3.54 | 0.0 | 16 | 80 / 80 | valid |
| v1 (20s) | LL→RR | `round_robin` | 8.0 | 160/160 | 160 | 7.313 | 94.4 / 109.9 / 113.1 | 2129.7 / 2233.7 / 2250.3 | 0.040 / 0.093 / 0.176 | 2.06 / 2.73 / 3.61 | 0.0 | 15 | 80 / 80 | valid |
| v1 (20s) | RR→LL | `round_robin` | 12.0 | 240/240 | 240 | 10.916 | 101.5 / 116.7 / 118.7 | 2300.8 / 2374.4 / 2384.8 | 0.038 / 0.089 / 0.546 | 1.76 / 2.20 / 3.21 | 0.0 | 26 | 120 / 120 | valid |
| v1 (20s) | RR→LL | `least_loaded` | 12.0 | 240/240 | 240 | 10.918 | 100.3 / 116.4 / 118.9 | 2290.7 / 2364.9 / 2377.7 | 0.039 / 0.073 / 0.094 | 1.81 / 2.25 / 2.95 | 0.0 | 26 | 118 / 122 | valid |
| v1 (20s) | LL→RR | `least_loaded` | 16.0 | 320/320 | 320 | 14.421 | 116.6 / 137.2 / 142.0 | 2735.9 / 2853.1 / 2870.0 | 0.041 / 0.072 / 0.107 | 3.14 / 4.24 / 6.77 | 0.0 | 44 | 156 / 164 | valid |
| v1 (20s) | LL→RR | `round_robin` | 16.0 | 320/320 | 320 | 14.427 | 115.3 / 140.7 / 144.3 | 2721.9 / 2898.9 / 2923.1 | 0.039 / 0.078 / 0.596 | 2.47 / 3.65 / 4.92 | 0.0 | 44 | 160 / 160 | valid |
| v1 (20s) | RR→LL | `round_robin` | 17.0 | 340/340 | 340 | 15.299 | 117.6 / 137.8 / 144.3 | 2771.7 / 2930.0 / 2949.3 | 0.037 / 0.072 / 0.216 | 3.43 / 6.01 / 8.17 | 0.0 | 47 | 170 / 170 | valid |
| v1 (20s) | RR→LL | `least_loaded` | 17.0 | 340/340 | 340 | 15.313 | 120.1 / 141.1 / 148.2 | 2784.7 / 2908.6 / 2928.6 | 0.039 / 0.071 / 0.096 | 3.06 / 6.42 / 9.31 | 0.0 | 47 | 168 / 172 | valid |
| v1 (20s) | LL→RR | `least_loaded` | 18.0 | 360/360 | 360 | 16.101 | 122.1 / 144.6 / 152.8 | 2831.9 / 3024.6 / 3045.5 | 0.041 / 0.086 / 0.288 | 3.30 / 5.88 / 9.18 | 0.0 | 52 | 177 / 183 | valid |
| v1 (20s) | LL→RR | `round_robin` | 18.0 | 360/360 | 360 | 16.174 | 120.5 / 142.4 / 147.4 | 2812.0 / 3016.7 / 3036.4 | 0.040 / 0.078 / 0.110 | 3.17 / 5.24 / 12.76 | 0.0 | 52 | 180 / 180 | valid |
| v1 (20s) | RR→LL | `round_robin` | 20.0 | 400/400 | 400 | 17.890 | 124.1 / 145.0 / 149.4 | 2897.5 / 3121.7 / 3149.9 | 0.038 / 0.075 / 0.107 | 3.72 / 6.21 / 10.90 | 0.0 | 60 | 200 / 200 | valid |
| v1 (20s) | RR→LL | `least_loaded` | 20.0 | 400/400 | 400 | 17.907 | 125.1 / 148.5 / 159.3 | 2911.4 / 3092.3 / 3127.0 | 0.041 / 0.086 / 0.803 | 4.05 / 9.90 / 17.66 | 0.0 | 60 | 197 / 203 | valid |
| v2 (10s) | RR→LL | `round_robin` | 20.0 | 200/200 | 200 | 16.390 | 116.1 / 137.9 / 142.6 | 2736.3 / 2798.7 / 2811.1 | 0.054 / 0.095 / 0.115 | 4.10 / 5.63 / 6.71 | 0.0 | 53 | 100 / 100 | valid |
| v2 (10s) | RR→LL | `least_loaded` | 20.0 | 200/200 | 200 | 16.295 | 120.2 / 143.0 / 148.1 | 2791.9 / 2889.8 / 2913.9 | 0.051 / 0.094 / 0.165 | 4.25 / 9.66 / 13.85 | 0.0 | 55 | 100 / 100 | valid |
| v2 (10s) | LL→RR | `least_loaded` | 28.0 | 280/280 | 280 | 22.253 | 145.6 / 187.2 / 210.2 | 3234.9 / 3528.2 / 3551.5 | 0.058 / 0.094 / 0.113 | 8.60 / 17.39 / 46.27 | 0.0 | 91 | 144 / 136 | valid |
| v2 (10s) | LL→RR | `round_robin` | 28.0 | 280/280 | 280 | 22.049 | 151.2 / 196.2 / 223.7 | 3270.8 / 3727.8 / 3753.8 | 0.058 / 0.096 / 0.936 | 11.74 / 15.54 / 32.66 | 0.0 | 94 | 140 / 140 | valid |
| v2 (10s) | RR→LL | `round_robin` | 32.0 | 320/320 | 320 | 24.237 | 174.8 / 265.0 / 302.5 | 3773.1 / 4694.5 / 4789.3 | 0.055 / 0.087 / 0.817 | 11.57 / 18.16 / 28.48 | 0.0 | 135 | 160 / 160 | valid |
| v2 (10s) | RR→LL | `least_loaded` | 32.0 | 320/320 | 320 | 24.472 | 178.0 / 250.1 / 280.5 | 3924.3 / 4564.8 / 4612.3 | 0.058 / 0.089 / 0.186 | 9.25 / 18.52 / 37.74 | 0.0 | 134 | 167 / 153 | valid |
| v2 (10s) | LL→RR | `least_loaded` | 34.0 | 340/340 | 340 | 25.621 | 193.0 / 268.0 / 298.5 | 4113.8 / 4902.1 / 4963.3 | 0.057 / 0.088 / 0.113 | 12.13 / 33.05 / 88.83 | 0.0 | 156 | 178 / 162 | valid |
| v2 (10s) | LL→RR | `round_robin` | 34.0 | 340/340 | 340 | 25.190 | 192.7 / 304.1 / 340.7 | 4092.4 / 5309.1 / 5367.9 | 0.054 / 0.093 / 0.209 | 17.27 / 42.01 / 70.39 | 0.0 | 161 | 170 / 170 | valid |
| v2 (10s) | RR→LL | `round_robin` | 36.0 | 360/360 | 360 | 26.834 | 199.6 / 313.3 / 354.7 | 4128.5 / 5227.8 / 5344.5 | 0.057 / 0.089 / 0.425 | 15.19 / 30.62 / 44.70 | 0.0 | 173 | 180 / 180 | valid |
| v2 (10s) | RR→LL | `least_loaded` | 36.0 | 360/360 | 360 | 26.872 | 216.0 / 306.0 / 349.3 | 4470.7 / 5401.7 / 5472.1 | 0.059 / 0.093 / 0.121 | 15.15 / 30.61 / 76.12 | 0.0 | 182 | 187 / 173 | valid |
| v2 (10s) | LL→RR | `least_loaded` | 44.0 | 440/440 | 440 | 29.507 | 316.1 / 1319.5 / 1437.6 | 5421.3 / 6393.6 / 6691.8 | 0.055 / 0.087 / 0.114 | 30.08 / 50.79 / 79.72 | 0.0 | 253 | 234 / 206 | valid |
| v2 (10s) | LL→RR | `round_robin` | 44.0 | 440/440 | 440 | 29.051 | 283.5 / 1110.6 / 1235.3 | 5063.3 / 6420.1 / 6751.2 | 0.053 / 0.089 / 0.326 | 29.26 / 44.16 / 54.99 | 0.0 | 247 | 220 / 220 | valid |

---

## 5. Selected Common Load Levels

Under the combined `1.0–44.0 RPS` evaluation (`experiments/raw/week8/pilot/combined_pilot_selection.json`):

| Load Level | Offered RPS | Run ID (`round_robin`) | Run ID (`least_loaded`) | Schedule SHA-256 | Justification |
| --- | ---: | --- | --- | --- | --- |
| **`low`** | `1.0` | `20261125T094219Z-w8-pilot-1rps-round-robin-b9c02ba-20028c` | `20261125T094431Z-w8-pilot-1rps-least-loaded-b9c02ba-569646` | `f12417a9076d3569b3d2ba9c85fa62172e3a1ed732b97b41ce020f67b7bd44ba` | Uncongested baseline (`TTFT p50 ~45–46 ms`, `E2E p50 ~1.73–1.77 s`, negligible backlog `1–2 / 20`). |
| **`medium`** | `4.0` | `20261125T094954Z-w8-pilot-4rps-round-robin-b9c02ba-2a693d` | `20261125T095142Z-w8-pilot-4rps-least-loaded-b9c02ba-407e07` | `54b01c6768b31967d771743d6224d814e7f692dcb06b109ec0244b97b34da1b4` | Tested rate closest to the log-space geometric midpoint `sqrt(1.0 * 28.0) = 5.292 RPS` (`TTFT p50 ~90 ms`, `E2E p50 ~1.97–1.98 s`, backlog `7–8 / 80`). Note: `8.0 RPS` is also validated as a slightly higher mid-range reference (`2129 ms` E2E). |
| **`near_saturation`** | `28.0` | `20261125T112407Z-w8-pilot-28rps-round-robin-2621c80-7179f6` | `20261125T112223Z-w8-pilot-28rps-least-loaded-2621c80-02d80e` | `bd186d5578c3887cb9d03085dbd15c5ef82a522333cdc3f10f15687f1223e69d` | Highest tested rate before `32.0 RPS` crosses `0.35` backlog (`0.422/0.419`), `34.0–36.0 RPS` breach `2.5 ×` baseline `E2E p95` (`>5002 ms`), and `44.0 RPS` exhibits normalized throughput saturation (`29.1–29.5` vs `36.7 RPS` expected) and a `1.11–1.32 s` TTFT p95 queue spike. *(Within `v2` alone relative to `20.0 RPS`, `32.0 RPS` was the last rate before `34.0 RPS` backlog > 0.43; for the full 54-cell campaign, `28.0 RPS` provides safer stability margin across longer windows and heterogeneous/burst profiles.)* |

---

## 6. Descriptive RR vs LL Observations (Single-Repetition Pilot Only)

- **Throughput**: Across all 15 rate pairs (`1.0–44.0 RPS`), achieved `completed_drain_rps` was nearly identical between RR and LL (within `<= 0.45 RPS` everywhere).
- **Worker balance**: RR always achieved exact 50/50 split (`160/160`, `220/220`), whereas LL exhibited modest asymmetry from `12.0 RPS` onward (e.g. `118/122` at 12 RPS, `167/153` at 32 RPS, `234/206` at 44 RPS) as fast-completing requests released reservations slightly asymmetrically across replicas.
- **Latency regime**: On homogeneous steady prompts (`max_tokens=64`, `n=1` repetition per rate), RR and LL showed closely overlapping E2E and TTFT up to `32.0 RPS`. At `34.0–44.0 RPS`, single-run jitter appeared in both directions (`34 RPS` E2E p95: RR `5309 ms` vs LL `4902 ms`; `36 RPS` E2E p50: RR `4128 ms` vs LL `4471 ms`; `44 RPS` E2E p50: RR `5063 ms` vs LL `5421 ms`, TTFT p95: RR `1111 ms` vs LL `1319 ms`).
- **No policy conclusion**: Because the pilot ran `1` repetition per rate on uniform prompts (`steady` profile only), no policy winner claim is warranted; multi-repetition controlled runs across `steady`, `burst`, and `heterogeneous` profiles are left for Member B's 54-cell campaign.

---

## 7. Remaining Work for Member B

1. Production `benchmark/config.py` (`schema_version=2`) and `benchmark/workload.py` (`load_workload(..., schema_version=2)` + `select_scheduled_entries(...)`).
2. Production `benchmark/artifacts.py` (`write_schedule`, `append_request`, `write_before_metrics`, W8 `environment.json` extensions) and `benchmark/summary.py` (W8 summary v2 formulas matching `contract_w8.md` section 7).
3. Authoring final `steady` / `burst` / `heterogeneous` workloads & manifests (`low=1.0`, `medium=4.0`, `near_saturation=28.0 RPS` — note for longer windows at `28.0 RPS`, keep `request_count` below `InMemoryTraceStore(max_size=500)` or ensure trace polling drains faster than `500` concurrent arrivals).
4. CLI v2 integration (`benchmark/__main__.py`), `infra/scripts/verify.py --target w8 --no-gpu`, analysis/plot scripts, and executing the 54-cell dual-T4 campaign.
