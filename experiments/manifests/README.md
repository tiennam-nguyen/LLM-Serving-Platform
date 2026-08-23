# Experiment Manifests

This directory stores experiment manifest files (e.g., YAML / JSON) that completely specify an experimental run.

---

## Required Manifest Fields

Each manifest documents:
- `experiment_id`: Unique identifier for the experiment run.
- `git_commit`: SHA of the codebase at execution time.
- `timestamp`: UTC timestamp of execution.
- `hardware`: GPU model(s), count, VRAM, node IPs.
- `model`: Exact HuggingFace model ID or local weights path and quantization settings.
- `routing_policy`: Policy name (`round_robin`, `least_loaded`, etc.) and hyperparameters.
- `workload`: Reference to workload trace or synthetic generator settings.
- `random_seed`: Seed used for reproducible replay.
