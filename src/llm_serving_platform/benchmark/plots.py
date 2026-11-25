"""Export scientific plots from newly derived analysis, with no server/GPU dependencies."""

from __future__ import annotations

import json
from pathlib import Path


def plot_analysis(analysis_path: Path, *, output_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    report = json.loads(analysis_path.read_text(encoding="utf-8"))
    if output_dir.exists():
        raise FileExistsError("Choose a new figures directory")
    rows = [
        s for s in report["runs"] if s["validation"]["valid"] and s["run_status"] == "completed"
    ]
    if not rows:
        raise ValueError("No complete, valid runs to plot")
    output_dir.mkdir(parents=True)
    outputs = []
    label = (
        "Controlled comparison"
        if report["eligible_for_policy_comparison"]
        else "Descriptive only; comparison gate not passed"
    )
    label += "; " + ", ".join(sorted({s.get("execution_mode", "unverified") for s in rows}))
    for profile in sorted({s["workload_profile"] for s in rows}):
        selected = [s for s in rows if s["workload_profile"] == profile]
        for metric in ("ttft", "e2e"):
            fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
            for ax, percentile in zip(axes, (50, 95, 99)):
                for policy in ("round_robin", "least_loaded"):
                    points = sorted(
                        [s for s in selected if s["policy"] == policy],
                        key=lambda s: s["rates"]["offered_rps"],
                    )
                    points = [
                        s for s in points if s["latency_ms"][f"{metric}_p{percentile}"] is not None
                    ]
                    ax.scatter(
                        [s["rates"]["offered_rps"] for s in points],
                        [s["latency_ms"][f"{metric}_p{percentile}"] for s in points],
                        label=policy,
                    )
                ax.set(
                    xlabel="Offered requests/s",
                    ylabel="Milliseconds",
                    title=f"{metric.upper()} P{percentile}",
                )
                ax.grid(alpha=0.25)
            axes[0].legend()
            fig.suptitle(f"{profile}: {label}; each point is one run")
            for suffix in ("png", "svg"):
                path = output_dir / f"{profile}_{metric}.{suffix}"
                fig.savefig(path, dpi=160)
                outputs.append(path)
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
        for policy in ("round_robin", "least_loaded"):
            points = [s for s in selected if s["policy"] == policy]
            ax.scatter(
                [s["rates"]["offered_rps"] for s in points],
                [s["rates"]["completed_window_rps"] for s in points],
                label=policy + " window",
            )
            ax.scatter(
                [s["rates"]["offered_rps"] for s in points],
                [s["rates"]["completed_drain_rps"] for s in points],
                marker="x",
                label=policy + " drain",
            )
        ax.set(
            xlabel="Offered requests/s", ylabel="Completed requests/s", title=f"{profile}: {label}"
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        path = output_dir / f"{profile}_throughput.png"
        fig.savefig(path, dpi=160)
        outputs.append(path)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(max(7, len(selected) * 0.25), 4.5), constrained_layout=True)
        bottoms = [0] * len(selected)
        for worker in sorted({w for s in selected for w in s["worker_distribution"]}):
            counts = [s["worker_distribution"].get(worker, 0) for s in selected]
            ax.bar(range(len(selected)), counts, bottom=bottoms, label=worker)
            bottoms = [a + b for a, b in zip(bottoms, counts)]
        ax.set_xticks(
            range(len(selected)),
            [
                f"{s['policy']}\n{s['comparison']['load_level']}\nr{s['comparison']['repetition']}"
                for s in selected
            ],
            fontsize=6,
        )
        ax.set(
            ylabel="Observed requests", title=f"{profile}: worker distribution (one bar per run)"
        )
        ax.legend(fontsize=8)
        path = output_dir / f"{profile}_workers.png"
        fig.savefig(path, dpi=160)
        outputs.append(path)
        plt.close(fig)
    return outputs
