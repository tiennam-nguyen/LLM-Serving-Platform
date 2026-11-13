# Week 5: two-machine LAN PoC

## A. Purpose and exit gate

Run two physical Linux machines, one independent vLLM replica per RTX 2080,
with a Gateway on Host A. Demonstrate remote endpoints, health over LAN,
Round-Robin, failure detection, exclusion, manual restart and recovery.
The operator runs and stops the processes and retains the evidence on both hosts.

**This is NOT the performance benchmark topology. Do not draw latency/throughput
policy conclusions from this LAN PoC.** No Least-Loaded, load accounting,
cache-aware/SLO routing, distributed KV or cross-node tensor parallelism is included.
TP is explicitly 1. The model stays `Qwen/Qwen2.5-0.5B-Instruct`.

Preparation is distinct from acceptance: CPU fixtures cannot prove physical GPU
identity or LAN behavior. Complete section S on the actual lab machines.

## B. Expected topology

```text
Host A                            Host B
RTX 2080                          RTX 2080
worker A :8000 <------ LAN ------> worker B :8000
    |                                 |
    +----- Gateway :8080 on A ---------+
           RoundRobinRouter
           Web UI :3000 (optional)
```

Only HTTP crosses hosts. Each worker loads a full model independently. The
Gateway's canonical configuration remains `WORKER_ENDPOINTS`; `HOST_A_IP` and
`HOST_B_IP` are launcher/runbook helpers, not new application settings.
Use a trusted lab LAN: these lab endpoints have no authentication or TLS.
Keep port access limited to the intended lab peers; do not expose them to the Internet.

## C. Before installation: inventory both machines

Run on **each host**, record which is A/B and match the physical GPU UUID:

```bash
nvidia-smi
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv
uname -a
cat /etc/os-release
python3 --version
hostname
ip -br -4 addr
ip route
date -Ins
timedatectl status
```

Record GPU name/UUID/VRAM, driver, OS/kernel, available Python, NIC/subnet,
default route and time synchronization state. Clock accuracy helps compare
host logs; Gateway trace durations use its own monotonic clock. Do not casually
upgrade lab drivers. If the installed serving wheel needs a different driver,
stop at that compatibility gate and consult the lab administrator.

## D. Clone the exact branch on both hosts

All commands below start in the repository root unless stated otherwise.

```bash
git clone https://github.com/tiennam-nguyen/LLM-Serving-Platform.git
cd LLM-Serving-Platform
git switch feat/week5-lan-poc
git rev-parse HEAD
git status --short
mkdir -p experiments/raw/week5_lan
git rev-parse HEAD > experiments/raw/week5_lan/revision.txt
bash infra/scripts/week5_lab_preflight.sh > experiments/raw/week5_lan/preflight-before.txt 2>&1
```

Compare full SHAs on A and B; they must match. Before generating evidence,
`git status --short` should be empty. The preflight is read-only and reports
missing optional tools. A failing NVIDIA inventory or installed PyTorch without
CUDA access produces a nonzero exit; fix that gate before proceeding. Missing
Python packages before installation are warnings. Run it again after installation.

## E. Environment setup

Prefer a clean Python **3.12** environment on both machines. Python 3.11 is also
within the project's declared range. Use the same installation path on A and B.

### Preferred: uv and the checked-in lock

If `uv` is not installed, obtain it via your lab's approved installation method
([official uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/)).
Do not install a system driver or system-wide Python packages from a launcher.

```bash
uv --version
uv venv --python 3.12 .venv
uv sync --locked --extra gpu --extra dev
source .venv/bin/activate
python --version
```

The canonical GPU extra is now `gpu = ["vllm"]`: vLLM supplies its own supported
Transformers range. The minimal correction preserves the lock's package versions,
including **vLLM 0.7.3**, PyTorch 2.5.1/CUDA 12.4 wheels and Transformers 4.48.3.
This is intentionally not a wholesale package upgrade. Do not use bare `uv run`
after this sync: a sync without extras can remove the GPU extra. The launchers
use the activated environment's `python` directly.

### Fallback: standard venv and pip

Use this alternative instead of uv when necessary:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[gpu,dev]'
python -m pip check
```

Pip does **not** consume `uv.lock`; this resolves a fresh serving environment and
can select a newer CUDA runtime with different driver requirements. Compare and
retain actual versions on both machines. Do not install arbitrary independent
Torch/Transformers versions to work around an error. To replay a deliberately
chosen serving version, constrain vLLM as part of the same resolution, for example
`python -m pip install -e '.[gpu,dev]' 'vllm==0.33.0'` only after checking the
lab's compatibility. **vLLM 0.33.0 was successfully used in the earlier dual-T4
validation**, with Python 3.12.16, PyTorch 2.14.0+cu130, CUDA 13.0 and driver
580.178.04. That evidence is on `exp/kaggle-dual-t4-real-replicas`, at
`docs/runbooks/kaggle_dual_t4.md`; it does not guarantee compatibility with the
lab RTX 2080 driver, and the 3B experiment model is not the Week 5 model.

The [versioned vLLM 0.7.3 GPU installation guide](https://docs.vllm.ai/en/v0.7.3/getting_started/installation/gpu/)
describes that wheel's CUDA/PyTorch coupling. For a freshly resolved runtime,
consult its matching version's installation guide rather than assuming the
[current guide](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
applies to the locked version.

Capture the actual environment before interpreting startup failures:

```bash
python -m pip --version  # pip may be absent in a uv-created environment
python - <<'PY'
import sys
from importlib.metadata import version
import torch
print('Python:', sys.version)
for package in ('torch', 'vllm', 'transformers'):
    print(package, version(package))
print('torch CUDA:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
PY
bash infra/scripts/week5_lab_preflight.sh > experiments/raw/week5_lan/preflight-installed.txt 2>&1
# Inspect this file and its exit status; do not proceed on a GPU/runtime FAIL.
cat experiments/raw/week5_lan/preflight-installed.txt
```

`nvidia-smi` reports the driver and its CUDA support; `torch.version.cuda` reports
the installed Torch build. They are different observations. Do not infer wheel
compatibility from a system `nvcc` version alone.

## F. Pre-download and prewarm the model

With Internet available, in the activated environment on **each host**:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download('Qwen/Qwen2.5-0.5B-Instruct'))
PY
```

For offline machines, copy a complete authorized Hugging Face cache (including
snapshot files and symlink targets) using the lab's approved shared storage.
Set `HF_HOME` to that cache in `.env.week5` and `HF_HUB_OFFLINE=1`; keep
`MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct` so the served/requested name agrees.
Check permissions and model files on both hosts before startup. Never commit
the cache. First startup plus the baseline's small streaming requests prewarms
the engines; startup/download time is not a routing performance measurement.

## G. Fill `.env.week5`

```bash
cp .env.week5.example .env.week5
${EDITOR:-nano} .env.week5
```

Replace **only `HOST_A_IP` and `HOST_B_IP`** to start; optionally tune documented
worker values. Use the same A/B identities on both hosts. Select the physical
LAN NIC reachable by the peer, excluding loopback, Docker, VPN and disconnected
interfaces. After loading the file, `ip route get "$HOST_B_IP"` on A (and the
reverse on B) shows the chosen interface/source address. Confirm that source
address matches the intended NIC. The two host IPs must differ.

The launchers source this trusted Bash file; do not source an untrusted file.
File values override inherited variables. Pass another trusted file as the
first argument to either launcher if needed. Without a file, exported variables
and documented defaults apply. Do not place secrets in the template.

On A, the Gateway launcher builds this **single comma-separated string**:

```text
worker-vllm-a=http://HOST_A_IP:8000,worker-vllm-b=http://HOST_B_IP:8000
```

Alternatively export `WORKER_ENDPOINTS` containing two explicit `id=http://host:port`
entries. The launcher checks it without requiring helper IPs. If IDs/ports differ
from the template, pass matching `--worker-a-id`, `--worker-b-id` and URL options
to the verifier. Keep the same model name on workers and verifier.

## H. Start worker A (Host A, terminal 1)

```bash
source .venv/bin/activate
bash infra/scripts/start_worker_lab.sh
```

Leave it visible; wait until startup completes. It prints FP16, memory fraction
0.6, context 2048, swap 1 GiB, GPU 0 and TP=1. In another terminal on A:

```bash
set -a; source .env.week5; set +a
curl -i --max-time 5 http://127.0.0.1:8000/health
curl -i --max-time 5 "http://$HOST_A_IP:8000/health"
ss -ltnp | grep ':8000'
```

Require HTTP 200 for both and bind `0.0.0.0:8000`, not only localhost.

## I. Start worker B (Host B, terminal 1)

```bash
source .venv/bin/activate
bash infra/scripts/start_worker_lab.sh
```

In a second terminal on B:

```bash
set -a; source .env.week5; set +a
curl -i --max-time 5 http://127.0.0.1:8000/health
curl -i --max-time 5 "http://$HOST_B_IP:8000/health"
ss -ltnp | grep ':8000'
```

Again require HTTP 200 and wildcard binding before continuing.

## J. Cross-host connectivity gate

From A (load `.env.week5` in this terminal first):

```bash
curl -i --connect-timeout 3 --max-time 5 "http://$HOST_B_IP:8000/health"
```

From B:

```bash
curl -i --connect-timeout 3 --max-time 5 "http://$HOST_A_IP:8000/health"
```

Both must return HTTP 200. Local health works + remote fails means a
bind/network/firewall/ACL/NIC problem. It is not a routing-policy diagnosis.
Inspect `ss`, `ip route get`, the selected NIC and peer addresses. Ask the lab
administrator for narrowly scoped peer-to-peer TCP 8000 access if required;
never disable firewalls globally. On proxy-configured hosts, set `NO_PROXY` for
localhost and both lab IPs in `.env.week5` (or remove proxy variables from the
service environment). Gateway httpx honors proxy settings; the verifier uses
direct HTTP. **Do not modify routing code until direct HTTP connectivity between
hosts works.**

## K. Start Gateway on Host A (terminal 2)

```bash
source .venv/bin/activate
bash infra/scripts/start_gateway_lab.sh
```

It rejects placeholders, pins live mode and Round-Robin, and prints two explicit
worker endpoints. In terminal 3 on A:

```bash
source .venv/bin/activate
set -a; source .env.week5; set +a
curl --max-time 5 http://127.0.0.1:8080/health
curl --max-time 5 http://127.0.0.1:8080/control/v1/workers
```

Require both IDs `HEALTHY`. Only one Gateway process is used; do not pass uvicorn
`--workers` or `--reload` during the demonstration. The registry and traces are
in memory. `/health` actively probes the pool, while the control API reads cached
state; the verifier polls control state before `/health` to observe the background
health monitor rather than force the transition itself.

## L. Baseline verification (Host A, terminal 3)

Keep other inference clients/UI requests idle during all three phases so the
strict sequential Round-Robin observation is meaningful.

```bash
python infra/scripts/verify_week5_lan.py \
  --phase baseline --gateway-url http://127.0.0.1:8080 \
  --worker-a-url "http://$HOST_A_IP:8000" --worker-b-url "http://$HOST_B_IP:8000" \
  --output experiments/raw/week5_lan/baseline.json
```

Exit 0 and `passed: true` require both direct health checks, exact configured
A/B IDs/endpoints in the registry, both healthy, four successful nonempty
streaming requests, matching request/header/trace worker IDs, completed traces
and all six milestones. Expect A B A B **or B A B A**, depending on the valid
starting cursor. The script reuses Week 4's streaming and cycle helpers.

## M. Failure demonstration

On **Host B terminal 1**, press Ctrl+C and wait for worker shutdown. Confirm B's
port 8000 is no longer listening. Do not stop the Gateway or worker A.

On A, terminal 3:

```bash
python infra/scripts/verify_week5_lan.py \
  --phase worker-b-down --gateway-url http://127.0.0.1:8080 \
  --worker-a-url "http://$HOST_A_IP:8000" --worker-b-url "http://$HOST_B_IP:8000" \
  --output experiments/raw/week5_lan/worker-b-down.json
```

Expect direct B unreachable/non-200, A HEALTHY, B UNHEALTHY, Gateway alive and
four new requests assigned only to A. Registry polling is bounded (default
30 seconds, `--wait-timeout-s` changes it). At interval 2 seconds/timeout 1 second,
allow several seconds for detection. Requests sent during that detection window
or already in flight can fail: there is no automatic retry, migration or drain
guarantee. The verifier waits for exclusion before sending the new requests.

## N. Recovery demonstration

On **Host B terminal 1**, restart the same command and wait for readiness:

```bash
bash infra/scripts/start_worker_lab.sh
```

On A, terminal 3:

```bash
python infra/scripts/verify_week5_lan.py \
  --phase recovered --gateway-url http://127.0.0.1:8080 \
  --worker-a-url "http://$HOST_A_IP:8000" --worker-b-url "http://$HOST_B_IP:8000" \
  --output experiments/raw/week5_lan/recovered.json
```

Require B's direct health 200, both registry records HEALTHY, B back in the
candidate set and an alternating four-request cycle. No Gateway restart should
be necessary. Per-request total bounds default to 90 seconds
(`--request-timeout-s`); timeouts are failures, never an infinite wait.

## O. Optional Web UI: backend must be GREEN first

The frontend normally defaults to `127.0.0.1:8080`. A remote browser's localhost
is the browser's own machine. On A, after all backend phases pass:

1. Add to A's `.env.week5` and restart the Gateway:
   `CORS_ALLOWED_ORIGINS=http://HOST_A_IP:3000,http://localhost:3000,http://127.0.0.1:3000`
   replacing `HOST_A_IP` with the actual address. Use exact browser origins, no wildcard.
2. In a new terminal on A (use the Node version allowed by `apps/web/package.json`):

```bash
set -a; source .env.week5; set +a
export NEXT_PUBLIC_GATEWAY_URL="http://$HOST_A_IP:8080"
export NEXT_PUBLIC_CONTROL_API_URL="http://$HOST_A_IP:8080"
cd apps/web
npm ci
npx next dev --hostname 0.0.0.0 --port 3000
```

Open `http://HOST_A_IP:3000` and `/workers`, substituting the address. Restart
Next after changing public environment values. Permit only intended LAN clients
to reach A's 8080/3000 if firewall rules require administrator assistance.
Do not use the UI to diagnose a broken lower-level LAN connection.

## P. Evidence collection

Each host keeps its own `experiments/raw/week5_lan/` directory; retain separately
as Host A and Host B when copying to approved storage. Never overwrite one host's
inventory with the other's. On each host:

```bash
git rev-parse HEAD > experiments/raw/week5_lan/revision.txt
nvidia-smi > experiments/raw/week5_lan/nvidia-smi.txt
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv \
  > experiments/raw/week5_lan/gpu.csv
ip -br -4 addr > experiments/raw/week5_lan/interfaces.txt
ip route > experiments/raw/week5_lan/routes.txt
bash infra/scripts/week5_lab_preflight.sh > experiments/raw/week5_lan/preflight-final.txt 2>&1
# With uv (also works for a uv environment without pip):
uv pip freeze > experiments/raw/week5_lan/packages.txt
# For a pip-managed environment instead: python -m pip freeze > .../packages.txt
```

On A, retain all three verifier JSON files. Each contains UTC timestamp, Git SHA,
phase, worker URLs/IDs, registry snapshot, request IDs, worker assignments,
six-milestone trace summaries, pass/fail and failure details. Generated content
and raw prompts are omitted. To save a separate registry snapshot after each phase:

```bash
curl --fail --max-time 5 http://127.0.0.1:8080/control/v1/workers \
  > experiments/raw/week5_lan/baseline-registry.json
# Repeat with worker-b-down-registry.json and recovered-registry.json at those phases.
```

Use different output filenames if retrying a failed phase to preserve diagnosis.
Logs, model caches and weights stay ignored. `.env.week5` is ignored; its example
is tracked. Small evidence files may be untracked under existing rules: inspect
`git status --short` and `git check-ignore <path>` before staging. Raw LAN evidence
can contain private addresses, hostnames and GPU UUIDs: retain locally or sanitize
deliberately before sharing; do not commit secrets, huge logs, caches or weights.

## Q. Troubleshooting by layer

| Symptom | Next checks / decision |
| --- | --- |
| vLLM does not start | Confirm activated Python 3.12/3.11 and installed vLLM; capture full terminal error and installed versions; distinguish download, import, CUDA and memory failures before changing dependencies. |
| CUDA/driver error | `nvidia-smi` fails -> lab GPU/driver access gate. Torch CUDA unavailable -> compare wheel runtime, driver and GPU visibility; ask administrator before changing drivers. Check the matching vLLM version's GPU guide. |
| GPU OOM | Keep 0.5B FP16; inspect other processes with `nvidia-smi`. First reduce `GPU_MEMORY_UTILIZATION` and `MAX_MODEL_LEN` in `.env.week5`, e.g. 0.5 / 1024, restart and record changes. Too little cache can itself prevent startup; inspect the exact error. Do not introduce quantization/new runtimes without evidence. |
| Local health OK, remote fails | Check wildcard bind, peer IP, selected NIC/route and narrow firewall/ACL access. Fix direct HTTP before routing code. |
| Both direct health OK, Gateway sees one unhealthy | Compare printed `WORKER_ENDPOINTS` with actual addresses/ports and registry URLs; check Gateway's proxy/NO_PROXY environment and health timeout. Check from A, not only worker-local terminals. |
| Gateway works, UI fails | Browser must use A's LAN URL; check Next public env, exact CORS origin, browser network errors and ports 3000/8080. Restart Next/Gateway after corresponding env changes. |
| Wrong LAN IP/NIC | `ip -br -4 addr`, `ip route get <peer-IP>`; choose the source address of the reachable physical LAN interface; update both files and restart Gateway if endpoints changed. |
| Port occupied | `ss -ltnp` filtered to the port; identify the owner. Stop only your known process via its terminal. If changing worker port, update launcher and Gateway endpoint/verifier URLs together. |
| Model unavailable/offline | Check complete HF cache, snapshot symlink targets, permissions and `HF_HOME`; online pre-download on both hosts or approved shared cache. Same served model name on workers and verifier. |
| Verifier fails cycle/trace check | Stop other inference clients; examine saved failure JSON and Gateway terminal. A valid shifted RR phase is accepted. A missing/completed-error trace must not be relabeled as a pass. |

The read-only preflight never changes drivers, routes, clock, firewall or packages.
Resolve infrastructure blockers at their layer before changing application code.

## R. Cleanup

Press Ctrl+C in Gateway, worker A, worker B and optional Web UI terminals. Wait
for shutdown; the launchers `exec` the processes so terminal signals reach them.

```bash
ss -ltnp | grep -E ':(8000|8080|3000)\b'
nvidia-smi
```

No matching owned listener is expected (grep exits 1 when none remain). Confirm
your vLLM GPU processes exited. Never use broad `killall python` on shared hosts.
Retain evidence and configuration locally for diagnosis; do not delete shared caches.

## S. Definition of Done (actual lab run)

- [ ] 2 physical hosts identified
- [ ] both RTX 2080 visible
- [ ] both hosts use same Git revision
- [ ] worker A local health passes
- [ ] worker B local health passes
- [ ] A -> B health over LAN passes
- [ ] B -> A health over LAN passes
- [ ] Gateway sees A/B HEALTHY
- [ ] live Round-Robin uses both machines
- [ ] inference streaming works through both workers
- [ ] request/worker trace correlation works
- [ ] stop B -> B becomes UNHEALTHY
- [ ] Gateway remains alive
- [ ] new requests route only to A
- [ ] restart B -> B becomes HEALTHY
- [ ] B returns to routing pool
- [ ] evidence artifacts saved
- [ ] known limitations documented

Final GREEN condition:

```text
A HEALTHY + B HEALTHY -> RR uses A/B
  -> stop B -> A HEALTHY + B UNHEALTHY -> new requests to A only
  -> restart B -> A HEALTHY + B HEALTHY -> B rejoins routing pool
```

Known limits: health is sampled, inference can fail between checks, there is no
in-flight retry guarantee, Gateway state is local/in memory and this is not a
performance benchmark or production security/reliability claim. Week 4 remains
complete. Week 5 completes only when this checklist has physical lab evidence;
Week 6 remains the primary multi-worker testbed plus Least-Loaded milestone.
