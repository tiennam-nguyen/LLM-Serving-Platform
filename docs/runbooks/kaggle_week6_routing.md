# Week 6 routing evidence on Kaggle

The launcher runs two independent vLLM replicas on one Kaggle node. GPU 0 serves
`worker-vllm-0` on port 8000 and GPU 1 serves `worker-vllm-1` on port 8001.
Each worker has one visible GPU and uses tensor parallel size 1. The Gateway
listens on loopback port 8080. All workload traffic goes through
`POST /v1/chat/completions` on the Gateway. The worker ports are used for health
and configured-model readiness.

## Run the no-GPU gate

```bash
python infra/scripts/verify.py --target w6 --no-gpu
python infra/scripts/verify_week6_e2e.py --no-gpu --save-evidence
```

The verifier uses controlled workers through the Gateway ASGI app. The default
output is `experiments/raw/week6/`. Use `--output-dir` for an independent run.
The test includes a real ASGI disconnect event and checks the terminal trace and
worker stream closure. The default homogeneous suite repeats concurrency 2, 4,
and 8 three times each.

## Run the dual-T4 gate

1. Provide Kaggle with a clean checkout of the exact Week 6 implementation
   revision, for example as a notebook input prepared from the local branch or
   from a repository ref you are authorized to transfer. Preserve Git metadata.
   Run from that checkout. Select **GPU T4 ×2** and enable internet for dependency
   and model downloads if the environment lacks them.
   For an unpublished local branch, create a Git bundle and add it as a
   private notebook input if you are authorized to transfer the source.

   ```bash
   git bundle create week6.bundle feat/w6-routing-evidence
   ```

   In the Kaggle notebook shell, replace `<dataset>` with the mounted input name.

   ```bash
   git clone --branch feat/w6-routing-evidence "/kaggle/input/<dataset>/week6.bundle" week6
   cd week6
   ```

2. From a notebook shell cell, confirm the two devices and run the launcher.
   Set `MODEL_NAME` to the configured model ID. The example uses the model from
   the earlier dual-T4 validation. Use your configured value when it differs.

```bash
nvidia-smi
REVISION="$(git rev-parse HEAD)"
MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"
python infra/kaggle/week6_routing_evidence/run.py \
  --bootstrap --revision "$REVISION" --model "$MODEL_NAME" \
  --output-dir /kaggle/working/week6/live
```

3. Download `/kaggle/working/week6/live/` and retain it under
   `experiments/raw/week6/live/`. Keep service logs local to the notebook unless
   needed for troubleshooting. The launcher uses the existing dual-T4 worker
   bootstrap, records GPU process ownership, and manages only its own process
   groups. It stops and restarts its worker on GPU 0 for the recovery scenario.
   Close other services on ports 8000, 8001, and 8080 before launching.

The launcher starts a fresh Gateway for each policy and controlled retry. Each
attempt checks that the two prior requests created the intended load and that the
probe trace actually captured the same unequal snapshot. It tries up to three
output budgets. A missed timing window is `INCONCLUSIVE`. A placement or cleanup
invariant violation is `FAIL`. The live phase also checks release on a request
with an invalid model ID. Live disconnect propagation remains inconclusive in
this launcher. The deterministic cancellation gate runs in no-GPU mode.

The default load matrix is bounded. Add `--extended` to include concurrency 16
once the baseline workload runs reliably. Review `validation.json` for gate
scope, `scenario_matrix.json` for per-request routing and Control API snapshots,
and `environment.json` for the exact revision and topology. Latency fields are
observations from these workloads.

## Probe a Gateway that is already running

```bash
python infra/scripts/verify_week6_e2e.py \
  --gateway-url http://127.0.0.1:8080 --model "$MODEL_NAME" \
  --policy least_loaded --save-evidence --output-dir /kaggle/working/week6/attached
```

Start a fresh, idle Gateway with two healthy workers and the specified policy
before running the attached probe. The attached mode checks one policy and one
controlled attempt. Fresh-policy tie evidence, restart retries, health recovery,
hardware proof, and the paired policy comparison are provided by the managed
launcher.
