# Week 8 controlled GPU benchmark — studies and evidence

## Latest result — 26 November 2026

Study v2 recorded 31/54 main runs after six separate conditioning runs and an
independent 18-run calibration. Its preregistered calibration precondition passed.
**30 main runs completed; run31 stopped at the old dispatch-lag guard; 23 did not start.**
Within the observed runs, **4,727/5,120 planned requests completed**; 0 errors, 0 cancellations and 0 missing traces. The complete 54-run plan contained 11,880 requests.

The complete methodology gate **did not pass**. Main results below are descriptive; they do not establish a policy ranking.

CPU/mock remains the default for pytest and the verifier. GPU execution is explicit
and remote; no local vLLM, CUDA workload or Docker was used. Measurement source a448b01
had 262 CPU tests and W7/W8 default gates passing before submission. The subsequent
implementation update is described separately below and has not been measured on GPU.

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

## Implementation update after v2 — CPU verified, GPU unverified

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
run summaries and numeric analyses without changing their legacy interpretation. New GPU execution has not been performed, and neither
v1 nor v2 is evidence of the new implementation on GPU. All original measurement source
commits and dates remain intact.

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
