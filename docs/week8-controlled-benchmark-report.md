# Week 8 controlled GPU benchmark — calibration evidence

## Result as of 26 November 2026

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

## Preregistered follow-up procedure

An independent v2 is planned with the same workloads, model, generation, load points
and 20% maximum band. It changes the preparation procedure before new observations:

1. Prefetch an exact model snapshot before GPU worker startup. Pin model and tokenizer
   to that revision and use cached offline workers.
2. Start both workers in parallel for every run, including the first.
3. Collect six conditioning runs (one medium-load pair per profile), explicitly excluded
   from calibration statistics and main comparisons. Preserve their raw data.
4. Record read-only GPU clock, temperature, power and utilization samples at1Hz for each run.
5. Run a new independent18-run calibration, freeze its tolerances under the original rule,
   and run54main trials only if the precondition passes.

CPU tests stay the default. Real GPU execution requires explicit `--gpu` or `--dual-t4`.
Kaggle account ownership is passed at submission time, with no personal account defaults
in shared source or configuration. No v2 result is claimed in this report yet.

The model-cache and revision controls follow the official
[Hugging Face download documentation](https://huggingface.co/docs/huggingface_hub/guides/download)
and [vLLM revision options](https://docs.vllm.ai/en/latest/cli/serve/).
GPU sampling uses the read-only query/loop options documented by
[NVIDIA-SMI](https://docs.nvidia.com/deploy/nvidia-smi/index.html).
