# Raw Experiment Data

This directory holds raw, unaggregated output files generated directly by the benchmark suite.

---

## File Types & Structure

- Per-request latency logs (JSONL / CSV) with timestamp, TTFT, ITL, total latency, and status code.
- Telemetry periodic snapshots (worker GPU memory, active requests, queue size).

*Note: Raw execution data files (`*.jsonl`, `*.parquet`, `*.csv`) are excluded from Git via `.gitignore` to prevent repository bloat. Keep raw data stored locally or in shared storage.*
