# Week 8 controlled GPU benchmark — studies and evidence

## Latest result — 27 November 2026 (UTC+7)

Replacement study v4 used clean source `e9a1e780a6686210bafcce5aa26fc0bbe5b25d14` on Kaggle dual-T4.
Its structured status is **completed** and methodology gate is
**False**. Main observed
90/90 runs, sending 13,865/19,800 planned requests; 13,865 completed,
0 errors, 0 cancellations and
0 missing traces. Detailed warnings, coverage and
eligibility are reported below. Execution completion does not establish a policy ranking.

V1/v2 retain their original outcomes; v3 was blocked during tokenizer preparation
before any measured request. None was rewritten as evidence of v4.
CPU/mock remains the default. The corrected helper passed all 313 CPU/mock tests,
Ruff and a separate real-Qwen-tokenizer CPU check. No local GPU/vLLM/Docker was used.

## Historical study v1 — 26 November 2026

Study v1 completed all 18 calibration runs on two real Tesla T4 GPUs. All 1,440 planned
requests were sent, observed and completed; there were no inference errors, cancellations
or missing traces. Raw validation passed for 18/18 runs and independent local recomputation
matched the remote numeric reports exactly.

The repeatability precondition **failed**. The main 54-run campaign was **not run**.
These calibration measurements cannot support a policy ranking or a claim that LL is faster.

| Item | Recorded value |
| --- | --- |
| Measurement source | `910b5d035aefd14c5cfc2ee4ba6bbc3ee6387fc0` (clean checkout) |
| Study start/end UTC | 2026-11-26T04:05:37.198169+00:00 / 2026-11-26T04:41:21.997871+00:00 |
| Model/topology | Qwen/Qwen2.5-3B-Instruct, two independent TP=1 replicas |
| GPU/driver | Two Tesla T4, 15360 MiB each, driver 580.178.04 |
| Python / vLLM / PyTorch | 3.12.16 / 0.33.0 / 2.14.0 |
| Calibration | 3 profiles × 2 policies × 3 repetitions at 4 offered RPS |
| Per-run input | 80 ordered requests over 20 seconds, seed42 |
| Generation / warm-up | temperature0, max_tokens64, top_p1; two separate warm-up requests |
| Cache procedure | Restart both workers and gateway before every run |
| Archive SHA-256 | `a6dbdb804f9d508e41cf08ab831b38c4b2d252aa6dad0bc4b9d974288d21fab0` |
| Local-only verification | CPU/mock; no local vLLM, CUDA workload or Docker |

The source rule, committed before observation, derives each primary tolerance as twice
the maximum relative deviation from the median across the six profile/policy groups.
It stops before main if any required tolerance exceeds20%. The source did not write a
frozen-tolerance file for v1 because this condition failed.

## Independent-run variation

Values below are the per-run medians for repetitions1/2/3, not pooled request percentiles.
“Required band” is the result of the declared formula; it was not accepted or relaxed.

| Profile / policy | TTFT P50 per run (ms) | E2E P50 per run (ms) | Required TTFT / E2E band |
| --- | --- | --- | --- |
| burst-medium / least_loaded | 116.145 / 116.770 / 117.750 | 825.615 / 824.925 / 826.975 | 1.679% / 0.329% |
| burst-medium / round_robin | 116.580 / 117.110 / 118.115 | 827.955 / 827.345 / 828.035 | 1.716% / 0.147% |
| heterogeneous-medium / least_loaded | 135.270 / 146.950 / 140.460 | 923.395 / 950.360 / 943.795 | 9.241% / 4.323% |
| heterogeneous-medium / round_robin | 137.360 / 135.775 / 138.485 | 936.400 / 932.900 / 948.520 | 2.308% / 2.589% |
| steady-medium / least_loaded | 94.875 / 99.540 / 99.270 | 771.745 / 813.340 / 803.950 | 8.855% / 8.012% |
| steady-medium / round_robin | 87.020 / 99.010 / 99.775 | 690.135 / 815.540 / 816.825 | 24.220% / 30.754% |

Steady RR is the failing group. Its first run differs from the next two even though the
ordered workload, settings and RR distribution (40/40) match. The original launcher
serialized the first two worker starts to avoid a cold model-download race, while later
runs started workers in parallel. This is a concrete procedural difference; v1 did not
record GPU clocks/temperature during inference, so a hardware/thermal cause is unproven.

## Retained evidence

- [Original preregistration](../experiments/raw/week8/controlled/20261126-910b5d0-v1/preregistration.json)
- [Original study result](../experiments/raw/week8/controlled/20261126-910b5d0-v1/study_report.json)
- [Raw run files](../experiments/raw/week8/controlled/20261126-910b5d0-v1/calibration/execution/runs)
- [Remote analysis](../experiments/processed/week8/controlled/20261126-910b5d0-v1/calibration/remote/analysis.json)
- [Independent recomputation](../experiments/processed/week8/controlled/20261126-910b5d0-v1/calibration/recomputed/analysis.json)
- [Import hashes and validation](../experiments/processed/week8/controlled/20261126-910b5d0-v1/import_verification.json)
- [Figures](../experiments/figures/week8/controlled/20261126-910b5d0-v1/calibration)

Original manifest paths, UTC dates, commit IDs and environment fields are retained.
The downloaded source archive remains in ignored local storage; its hash is recorded
above. Original A pilot/W6/W7 artifacts were not edited. New learning guides and the
collaboration contract remain outside this repository in `D:\dath`.

## Study v2 — independent preparation and calibration

Source and preparation were committed before new observations. Workloads, load points,
model, generation and the original 20% cap were retained. Preparation added a prefetched
model/tokenizer snapshot, offline workers, parallel startup for every run, six separate
conditioning runs and read-only GPU telemetry at 1 Hz. Conditioning measurements are
excluded from calibration tolerances and main policy comparisons.

| Item | Recorded value |
| --- | --- |
| Measurement source | `a448b018e622a75862d75cf400729e6527e2afa7`; dirty=False |
| Study start/end UTC | 2026-11-26T05:19:04.278085+00:00 / 2026-11-26T06:57:08.518319+00:00 |
| Study status / methodology gate | `failed` / `false` |
| Pinned model/tokenizer snapshot | `aa8e72537993ba99e69dfaafa59ed015b17504d1` |
| Archive SHA-256 | `369678153135787b5054cc8d3b245de16ec0fad79022a836e7bd4e10147bff55` |

Recorded stop: `RuntimeError: Main execution failed; partial raw preserved`.

| Phase | Runs observed / planned | Valid raw | Completed / planned requests |
| --- | --- | --- | --- |
| conditioning | 6 / 6 | 6 | 480 / 480 |
| calibration | 18 / 18 | 18 | 1,440 / 1,440 |
| main | 31 / 54 | 31 | 4,727 / 5,120 |

Remote calibration numeric results were independently reproduced exactly. Main stopped
before remote analysis; its partial raw was analyzed locally for descriptive reporting.
The request counts in this table cover observed runs, including the partial run31;
they do not include the 23 unstarted runs. Source archive members and imported copies
have SHA-256 records; the
analysis code hashes are recorded separately. All original dates, paths and source
commits remain intact.

### Frozen calibration tolerances

The same formula was applied: twice the largest relative deviation from the median
across six profile/policy groups. Tolerances were frozen before any main run.

| Primary metric | Frozen relative tolerance |
| --- | --- |
| `latency_ms.ttft_p50` | 4.828690% |
| `latency_ms.e2e_p50` | 4.053553% |
| `rates.completed_drain_rps` | 3.820424% |

Freeze time UTC: 2026-11-26T06:03:28.706088+00:00. The file records the preregistration, model cache, protocol and analysis hashes plus the conditioning
and calibration run IDs. Main results were not used to tune these values.

### Main measurements

Each cell is the median of the available per-run statistics. The Runs column shows
completed/observed runs: the last group has only one partial run and is not a valid
three-repetition comparison. Request samples are not pooled. RR = round_robin, LL = least_loaded.

| Profile | Load (RPS) | Policy | Runs | Completed requests | TTFT P50 ms | E2E P50 ms | E2E P95 ms | Drain RPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| burst | low (1) | RR | 3/3 | 60 | 53.645 | 1291.430 | 2359.073 | 1.000000 |
| burst | low (1) | LL | 3/3 | 60 | 52.910 | 1172.325 | 2357.081 | 1.000000 |
| burst | medium (4) | RR | 3/3 | 240 | 117.620 | 835.950 | 1083.176 | 4.000000 |
| burst | medium (4) | LL | 3/3 | 240 | 118.320 | 836.535 | 1081.475 | 4.000000 |
| burst | near_saturation (28) | RR | 0/1 | 167 | 320.620 | 1412.590 | 1779.293 | 8.350000 |
| steady | low (1) | RR | 3/3 | 60 | 49.565 | 934.220 | 1366.343 | 1.000000 |
| steady | low (1) | LL | 3/3 | 60 | 49.845 | 945.430 | 1236.305 | 1.000000 |
| steady | medium (4) | RR | 3/3 | 240 | 107.595 | 902.730 | 1216.130 | 3.797119 |
| steady | medium (4) | LL | 3/3 | 240 | 103.950 | 887.885 | 1218.709 | 3.793085 |
| steady | near_saturation (28) | RR | 3/3 | 1,680 | 123.805 | 1067.650 | 1580.671 | 26.765878 |
| steady | near_saturation (28) | LL | 3/3 | 1,680 | 119.745 | 1005.530 | 1396.897 | 26.704567 |

Matched policy pairs: 15/16. Repeatability groups passing: 7/11.
Protocol validation: `{"coverage_complete": false, "hash_matched": true, "predeclared": true}`.

| Repeatability group | Status | Failing primary checks |
| --- | --- | --- |
| w8-burst-low / least_loaded | failed | latency_ms.e2e_p50: deviation 11.568% > tolerance 4.054% |
| w8-burst-low / round_robin | failed | latency_ms.e2e_p50: deviation 15.388% > tolerance 4.054% |
| w8-burst-near_saturation / round_robin | insufficient_repetitions | See recorded condition/provenance checks |
| w8-steady-medium / round_robin | failed | latency_ms.ttft_p50: deviation 5.251% > tolerance 4.829% |

### Why Kaggle v2 reported ERROR

Run31 was burst / near_saturation / RR repetition1. The 20-second schedule contained
560 requests in groups of56 every two seconds. Only167 were sent; all167 completed
with zero inference errors and zero missing traces. The remaining393 were recorded
as not dispatched. Reconstructing HTTP overlap from the ledger gives a peak of56,
below the350 cap. Drain ended at6.029624 seconds, before the20-second window boundary.
The last sent sequence166 was scheduled for4000ms and dispatched at4249.332ms.
Sequence167 was also scheduled for4000ms. These facts identify the250ms dispatch-lag
guard as the stopping branch; the exact rejected attempt timestamp was not retained.
The old launcher raised RuntimeError after any failed main run, producing Kaggle ERROR.
This is not evidence that the GPU could not process28RPS, and the retained failure
does not show GPU quota exhaustion. The deeper source of scheduling delay was not profiled.

## Implementation update after v2 — observation semantics

Production replay now records late dispatch and continues the complete schedule, even
after the nominal arrival window. At the hard in-flight cap it records that request
as not dispatched with an explicit reason and proceeds to later arrivals, without
adding a client queue. Dispatch timestamps mark HTTP-client invocation, not network
arrival. Policy mismatch is failed; unavailable infrastructure is blocked; explicit
cancellation is cancelled. Missing traces and request outcomes remain visible.

Campaigns continue across slow runs and retain warning counts and timing metrics.
Unreliable calibration no longer prevents observation of main: no verified tolerance
is invented, and formal methodology acceptance remains false. The original A pilot
explicitly retains its strict execution mode; legacy raw keeps its original interpretation.

The additional realistic suite combines steady, irregular, finite spike and ramp arrivals
with short/medium/long prompts. Stress optionally adds mixed microbursts. Schedules are
materialized once and shared by RR/LL. Remote preparation uses the pinned model chat
tokenizer to fit input bands96–192/384–640/1024–1664 tokens including template overhead.
Generation remains temperature0/max_tokens64/top_p1; actual output may stop earlier.
Local CPU fixtures explicitly do not claim real-model token measurements.

### Tokenizer compatibility check before the replacement GPU run

The first new submission (v3, source `cb863de`) exposed a preparation bug:
Transformers 5.20.2 returns `BatchEncoding` by default from `apply_chat_template`.
Taking its length counts mapping keys, not input tokens. The helper now explicitly
requests `return_dict=False`; the regression mock reproduces the new mapping default.
A separate CPU environment loaded the actual pinned Qwen tokenizer without torch,
vLLM or model weights. Its fixture had 34 tokens versus two mapping keys; fitted
short/medium/long samples had 116/389/1040 tokens, within their respective bands.
The v3 study ended at 12:09:56 UTC on 26 November with `blocked` and
`ValueError: Tokenizer did not reach the requested input length`, before conditioning
or any measured request. Kaggle's outer status is `COMPLETE` because the script saved
its structured blocked report successfully. This is distinct from passing the study.
The [v3 report](../experiments/raw/week8/controlled/20261126-cb863de-v3/study_report.json),
[import verification](../experiments/processed/week8/controlled/20261126-cb863de-v3/import_verification.json)
and [real tokenizer CPU proof](../experiments/processed/week8/controlled/20261126-tokenizer-compatibility/real-tokenizer-verification.json)
retain that outcome. This CPU check does not establish GPU benchmark results.
See the [Transformers 5.20.2 tokenizer API](https://huggingface.co/docs/transformers/v5.20.2/en/main_classes/tokenizer).

These changes have 313 CPU/mock tests passing, plus Ruff and offline lock validation.
The CPU W8 gate passed 16 real HTTP runs, 168 measured requests and 48 figures; W7 also passed.
[CPU verification and code hashes](../experiments/processed/week8/controlled/20261126-observation-cpu/verification.json)
record the working tree actually checked. The new reader reproduced all 73 retained v1/v2
run summaries and numeric analyses without changing their legacy interpretation. Replacement GPU v4 is reported separately below. V1/v2 are not evidence of
the new implementation; v3 records a preparation block. All original source
commits and dates remain intact.

## Study v4 — retained execution and measurement quality

| Item | Recorded value |
| --- | --- |
| Measurement source | `e9a1e780a6686210bafcce5aa26fc0bbe5b25d14` (clean checkout) |
| Model / topology | Qwen/Qwen2.5-3B-Instruct @ `aa8e72537993ba99e69dfaafa59ed015b17504d1`; two independent TP=1 replicas |
| Study start / end UTC | 2026-11-26T12:19:54.548957+00:00 / 2026-11-26T17:00:28.229964+00:00 |
| Structured study status / methodology gate | completed / False |
| Suite / planned cases | stress: 10 conditioning + 30 calibration + 90 main |
| Arrivals | steady, irregular, finite spike, ramp, microburst; all mix input lengths |
| Nominal main loads / repetitions | 1/4/28 RPS over 20s; RR/LL ×3 independent repetitions |
| HTTP in-flight cap | 350, safety limit rather than configured concurrency |
| GPU/driver | GPU0 Tesla T4 15360MiB driver 580.178.04; GPU1 Tesla T4 15360MiB driver 580.178.04 |
| Python / vLLM / torch / Transformers | 3.12.16 / 0.33.0 / 2.14.0 / 5.20.2 |
| Cache controls | `{"workers_restarted_between_runs": true, "fresh_mock_gateway": false, "prefix_cache_disabled": false, "worker_startup": "parallel_prefetched", "model_cache_offline": true}` |
| Execution semantics / dispatch clock | observe_v1 / before_http_client_call |
| Archive SHA-256 | `4c8046d46a2efdb20c4242ae1eda392d9600905f9f81002d0a78d053cf7ceb6a` |

`prefix_cache_disabled=false` records that the launcher did not apply a disable
flag. Worker processes were restarted, but these controls do not establish that
within-run prefix reuse was disabled. No cause of the calibration variation was isolated.

Each RR/LL pair replays the same materialized workload bytes, including timestamps
and prompt ordering. At the client cap, some scheduled requests are not transmitted;
matching planned hashes does not imply identical delivered request sets. These are
client admission omissions, not gateway rejections or inference errors. Their counts
remain in planned totals and noncompletion rates; affected runs are ineligible for
comparison under the intended schedule. Prompt bands are 96–192, 384–640, 1024–1664 input tokens including
chat-template overhead, checked using the pinned Qwen tokenizer. Generation remains
temperature=0, max_tokens=64, top_p=1. Two warm-up requests per run are outside measured counts.

An independent offline CPU recount matched all 9,900 prepared workload rows across
the three phase copies (627 unique messages). Actual input lengths in this study
were 99/393/1044 tokens for short/medium/long. These are three discrete input lengths,
not uniformly sampled lengths throughout the allowed bands; ordering mixes them.

| Phase | Observed runs / planned | Completed / planned requests | Errors / missing trace | Raw valid | Remote numeric report reproduced |
| --- | --- | --- | --- | --- | --- |
| conditioning | 10/10 | 800/800 | 0/0 | 10 | n/a (no remote analysis) |
| calibration | 30/30 | 2400/2400 | 0/0 | 30 | True |
| main | 90/90 | 13865/19800 | 0/0 | 90 | True |

Main status counts: `{"completed": 90}`.
Main observed-policy checks: `{"passed": 90}`.
Main warning counts: `{"dispatch_late": 95, "in_flight_limit": 5935}`.
Maximum reconstructed simultaneous HTTP requests: **350**;
maximum dispatch lag: **324.286ms**.
The timestamp is immediately before HTTP-client invocation, not arrival at the gateway.
Warnings reflect generator/measurement limitations and remain in the results.
The tokenizer correction also has its own [full 313-test CPU result](../experiments/processed/week8/controlled/20261126-tokenizer-compatibility/pytest-junit.xml),
separate from the earlier observation-mode CPU gate evidence.
Main runs without both workers observed idle at the end:
**0**.
Runs whose dispatch exceeds 250ms: **14**; their final
statuses are retained in the raw summaries. Delays do not trigger the old hard-stop guard.

### Eligibility and calibration

Calibration uses twice the maximum relative deviation from each profile/policy
group's median, with the preregistered 20% maximum acceptance cap. Required bands
below are calculations, not automatically accepted tolerances.

| Metric | Required band | Frozen tolerance |
| --- | --- | --- |
| latency_ms.ttft_p50 | 61.354200% | not frozen |
| latency_ms.e2e_p50 | 14.544490% | not frozen |
| rates.completed_drain_rps | 4.266832% | not frozen |

Study warnings: `[{"type": "calibration_unreliable", "detail": "Calibration noise exceeds preregistered tolerance limit"}, {"type": "measurement_warnings", "phase": "main", "counts": {"dispatch_late": 95, "in_flight_limit": 5935}}, {"type": "methodology_gate_not_passed"}]`.
Main structurally valid runs: **90**;
individually comparison-eligible runs: **60**.
Pair matching: `{"True": 45}`.
Repeatability groups: `{"not_evaluated_no_predeclared_tolerances": 30}`.
Main comparison eligibility: **False**.

Protocol coverage and hash checks are `{"coverage_complete": true, "hash_matched": true, "predeclared": false}`.
Here `predeclared=false` is the reader's repeatability-tolerance gate: calibration
did not yield accepted tolerances, so protocol repeatability status is
`calibration_unreliable`. The workload design and policy order were not tuned from
main outcomes. All 45 planned RR/LL pairs matched their input/settings invariants,
while 30 high-load runs were ineligible because of cap omissions; 14 of those also
exceeded the dispatch-lag warning threshold. The 60 individually eligible runs do
not establish a whole-study comparison without accepted tolerances.

Ineligible reasons: `{"dispatch_late": 14, "in_flight_limit": 30}`.
These distinguish source/input matching from the ability to reproduce the scheduled
load and repeat the measurements. Unreliable calibration does not create a verified
tolerance; main observations continue with that limitation visible. No thresholds
were relaxed after main observation and no runs were removed to improve the result.

### Main descriptive medians

Rows are medians of per-run metrics across the observed repetitions, rather than pooled
request percentiles. They describe this synthetic workload and environment; completion
and matching inputs alone do not support a general policy ranking.
Timeout/error/cancellation cleanup has CPU/mock and HTTP-test coverage. The GPU
suite does not establish fault-injection coverage for outcomes it did not observe.

| Profile | Load | Policy | Runs | Completed/planned | TTFT P50 ms | E2E P50 ms | Drain RPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| burst_mixed | low | least_loaded | 3 | 60/60 | 212.335 | 1221.260 | 0.982 |
| burst_mixed | low | round_robin | 3 | 60/60 | 222.870 | 1242.005 | 0.981 |
| burst_mixed | medium | least_loaded | 3 | 240/240 | 599.945 | 2468.475 | 3.838 |
| burst_mixed | medium | round_robin | 3 | 240/240 | 637.015 | 2493.930 | 3.840 |
| burst_mixed | near_saturation | least_loaded | 3 | 1132/1680 | 15388.290 | 33394.790 | 7.844 |
| burst_mixed | near_saturation | round_robin | 3 | 1117/1680 | 15045.910 | 33414.130 | 7.710 |
| irregular_mixed | low | least_loaded | 3 | 60/60 | 196.385 | 1155.060 | 0.960 |
| irregular_mixed | low | round_robin | 3 | 60/60 | 158.140 | 1088.845 | 0.969 |
| irregular_mixed | medium | least_loaded | 3 | 240/240 | 348.825 | 1485.830 | 3.830 |
| irregular_mixed | medium | round_robin | 3 | 240/240 | 281.915 | 1462.980 | 3.796 |
| irregular_mixed | near_saturation | least_loaded | 3 | 1073/1680 | 14182.150 | 33576.935 | 7.695 |
| irregular_mixed | near_saturation | round_robin | 3 | 1073/1680 | 14544.305 | 33651.360 | 7.731 |
| microburst_mixed | low | least_loaded | 3 | 60/60 | 171.030 | 1259.775 | 1.000 |
| microburst_mixed | low | round_robin | 3 | 60/60 | 152.995 | 1181.445 | 1.000 |
| microburst_mixed | medium | least_loaded | 3 | 240/240 | 796.710 | 1584.550 | 4.000 |
| microburst_mixed | medium | round_robin | 3 | 240/240 | 758.145 | 1555.070 | 4.000 |
| microburst_mixed | near_saturation | least_loaded | 3 | 1058/1680 | 15130.080 | 34677.670 | 7.686 |
| microburst_mixed | near_saturation | round_robin | 3 | 1056/1680 | 15658.910 | 34043.580 | 7.568 |
| ramp_mixed | low | least_loaded | 3 | 60/60 | 174.985 | 1243.270 | 0.915 |
| ramp_mixed | low | round_robin | 3 | 60/60 | 179.355 | 1246.425 | 0.919 |
| ramp_mixed | medium | least_loaded | 3 | 240/240 | 294.595 | 1693.835 | 3.705 |
| ramp_mixed | medium | round_robin | 3 | 240/240 | 364.665 | 1783.765 | 3.675 |
| ramp_mixed | near_saturation | least_loaded | 3 | 1111/1680 | 12608.050 | 32930.020 | 7.428 |
| ramp_mixed | near_saturation | round_robin | 3 | 1098/1680 | 12772.480 | 33164.635 | 7.371 |
| steady_mixed | low | least_loaded | 3 | 60/60 | 153.065 | 1206.400 | 0.966 |
| steady_mixed | low | round_robin | 3 | 60/60 | 145.030 | 1148.410 | 0.965 |
| steady_mixed | medium | least_loaded | 3 | 240/240 | 220.935 | 1375.930 | 3.767 |
| steady_mixed | medium | round_robin | 3 | 240/240 | 229.270 | 1390.325 | 3.776 |
| steady_mixed | near_saturation | least_loaded | 3 | 1074/1680 | 14663.805 | 34186.035 | 7.697 |
| steady_mixed | near_saturation | round_robin | 3 | 1073/1680 | 14614.270 | 34468.090 | 7.648 |

### Inspect v4 evidence

- [Preregistration](../experiments/raw/week8/controlled/20261126-e9a1e78-v4/preregistration.json), [study report](../experiments/raw/week8/controlled/20261126-e9a1e78-v4/study_report.json)
- [Import checksums and independent recomputation](../experiments/processed/week8/controlled/20261126-e9a1e78-v4/import_verification.json)
- [Main raw](../experiments/raw/week8/controlled/20261126-e9a1e78-v4/main/execution/runs), [main numeric analysis](../experiments/processed/week8/controlled/20261126-e9a1e78-v4/main/recomputed/analysis.json)
- [Calibration numeric analysis](../experiments/processed/week8/controlled/20261126-e9a1e78-v4/calibration/recomputed/analysis.json)
- [Real-tokenizer CPU proof](../experiments/processed/week8/controlled/20261126-tokenizer-compatibility/real-tokenizer-verification.json)
- [Independent recount of imported workload tokens](../experiments/processed/week8/controlled/20261126-e9a1e78-v4/workload-tokenizer-verification.json)
- [Provider log/status verification](../experiments/processed/week8/controlled/20261126-e9a1e78-v4/provider_verification.json)
- [Git byte-preservation verification](../experiments/processed/week8/controlled/20261126-e9a1e78-v4/git_byte_verification.json)
- [Main figures](../experiments/figures/week8/controlled/20261126-e9a1e78-v4/main)

Recompute into a new output directory with CPU and the analysis extra:

```powershell
python -m llm_serving_platform.benchmark analyze --runs-root experiments/raw/week8/controlled/20261126-e9a1e78-v4/main/execution/runs --protocol experiments/manifests/week8/controlled/20261126-e9a1e78-v4/main/campaign.json --output-dir .cache/week8-replay-v4 --figures
```

### Limits and interpretation

The offered 1/4/28 RPS points came from the original steady pilot; those labels do not
prove saturation for burst or heterogeneous inputs. This is a short synthetic workload
on two TP=1 replicas on one host, not a multi-host capacity study. Low-load runs have
only 20 requests each, so P99 describes a small sample. Three repetitions do not establish
statistical significance or a general claim that either policy is always faster.

V1 and v2 are separate sessions with changed preparation. V2 telemetry cannot reconstruct
v1 clocks/temperature or isolate the cause of v1 variation. No v1 run was discarded and
the calibration formula and cap were not relaxed after observation.

### Reproduce and inspect

- [Preregistration](../experiments/raw/week8/controlled/20261126-a448b01-v2/preregistration.json)
- [Study result](../experiments/raw/week8/controlled/20261126-a448b01-v2/study_report.json)
- [Import hashes and independent verification](../experiments/processed/week8/controlled/20261126-a448b01-v2/import_verification.json)
- [Main raw](../experiments/raw/week8/controlled/20261126-a448b01-v2/main/execution/runs)
- [Main numeric analysis](../experiments/processed/week8/controlled/20261126-a448b01-v2/main/recomputed/analysis.json)

Recompute with CPU and the analysis extra installed, using a new output directory:

```powershell
python -m llm_serving_platform.benchmark analyze --runs-root experiments/raw/week8/controlled/20261126-a448b01-v2/main/execution/runs --protocol experiments/manifests/week8/controlled/20261126-a448b01-v2/main/campaign.json --output-dir .cache/week8-replay-v2 --figures
```

Kaggle ownership is supplied at submission time. There is no personal account default
in shared source or configuration. The learning guides and collaboration contract remain
outside the repository in `D:\dath`. Model-cache controls follow the official
[Hugging Face download documentation](https://huggingface.co/docs/huggingface_hub/guides/download)
and [vLLM revision options](https://docs.vllm.ai/en/latest/cli/serve/). Sampling uses the
read-only query/loop options documented by [NVIDIA-SMI](https://docs.nvidia.com/deploy/nvidia-smi/index.html).
