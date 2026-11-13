# Week 5 evidence directory

Physical lab evidence is **pending**. No baseline, outage or recovery result has
been claimed or fabricated here. Follow [the lab runbook](../../../docs/runbooks/week5_lan_poc.md)
to generate `baseline.json`, `worker-b-down.json` and `recovered.json` on Host A,
and retain separate host/GPU/runtime/network inventories from both machines.
The operator owns these local observations and their retention. Review private
addresses, hostnames and GPU UUIDs before sharing; do not commit raw logs or caches.

## Preparation verification receipt

Base: `main` at `db60212fadeaf71f8c0092a65dffd6c5629fa1d0`.
The commit containing this receipt identifies the prepared source revision.
The local verification environment was Windows build 26200, CPython 3.11.9,
Git Bash 5.2.37; no GPU packages were installed. CPU dependencies were resolved
with `uv pip install -e '.[ci]'` in ignored `.cache/week5-venv`:
FastAPI 0.162.6, httpx 0.28.1, uvicorn 0.54.0, pytest 9.2.0,
pytest-asyncio 1.5.0 and Ruff 0.17.5.

[RAN] Before edits, all four commands below passed: Ruff, 39 unit tests,
8 integration tests and the Week 4 mock lifecycle gate. After edits, all four
passed: Ruff, **47 unit tests**, **17 integration tests**, and the unchanged
Week 4 mock gate.

```bash
python infra/scripts/verify.py --target lint --no-gpu
python infra/scripts/verify.py --target unit --no-gpu
python infra/scripts/verify.py --target integration --no-gpu
python infra/scripts/verify.py --target w4 --no-gpu
bash -n infra/scripts/week5_lab_preflight.sh
bash -n infra/scripts/start_worker_lab.sh
bash -n infra/scripts/start_gateway_lab.sh
uv lock --check --offline
```

[RAN] All three Bash syntax checks and the lock check passed. An independent
TOML comparison against the base lock verified all **147 package identities,
versions and sources unchanged**; the GPU extra contains only `vllm`. The lock
delta matches the accepted Kaggle branch's minimal metadata correction.

[RAN] New tests exercise all three observation phases through the real Gateway,
background health monitor, Round-Robin router and vLLM HTTP adapters using an
in-process controlled HTTP transport. They cover shifted starting cursor,
exclusion/re-entry, saved JSON and eight negative evidence cases. Launcher
contracts use a stub process to inspect arguments/live mode and reject bad
configuration. Missing-tool preflight testing confirms an unavailable NVIDIA
inventory produces a clear nonzero failure.

The Windows sandbox stalled async baseline tests and blocked Git Bash startup;
the successful runs used the host-approved execution path outside that sandbox.
Local final command output is in ignored `.cache/week5-verification/`. Pytest
emits an existing warning about unset `asyncio_default_fixture_loop_scope`;
there were no test failures. Frontend files were unchanged.

[UNCHECKED] Linux GPU installation, actual RTX 2080 CUDA/driver compatibility,
physical two-host connectivity, real vLLM startup/streaming, firewall/NIC setup
and stop/restart behavior on lab hardware. The retained lock uses vLLM 0.7.3;
the accepted older dual-T4 experiment's separately resolved vLLM 0.33.0 is not
evidence for this topology. This preparation is not a performance benchmark.
