"""Run the reproducible held-out, paired policy benchmark."""

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from core.benchmark import paired_difference_intervals, run_seeded_benchmark, summarize_seeded_benchmark
from core.environment import HELDOUT_DOMAINS


APP_ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plot_metric(summary, metric, ylabel, destination, lower_is_better=False):
    rows = [r for r in summary if r["metric"] == metric]
    rows.sort(key=lambda r: r["estimate"], reverse=not lower_is_better)
    names = [r["method"] for r in rows]
    estimates = [r["estimate"] for r in rows]
    low = [r["estimate"] - r["ci95_low"] for r in rows]
    high = [r["ci95_high"] - r["estimate"] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.errorbar(range(len(names)), estimates, yerr=[low, high], fmt="o", capsize=4, color="#145a72")
    ax.set_xticks(range(len(names)), names, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Held-out benchmark: {ylabel} (95% hierarchical bootstrap CI)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=300, help="Common held-out candidate episodes per checkpoint")
    parser.add_argument("--base-seed", type=int, default=91000)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--output-dir", type=Path, default=APP_ROOT / "reports" / "seeded_benchmark")
    args = parser.parse_args()
    if args.episodes < 1 or args.bootstrap_samples < 100:
        parser.error("--episodes must be >= 1 and --bootstrap-samples must be >= 100")

    checkpoint_dir = APP_ROOT / "models" / "benchmark"
    checkpoints = sorted(checkpoint_dir.glob("*_seed*.pt"))
    rows = run_seeded_benchmark(checkpoint_dir, episodes=args.episodes, base_seed=args.base_seed)
    summaries = summarize_seeded_benchmark(rows, bootstrap_samples=args.bootstrap_samples)
    paired = paired_difference_intervals(rows, reference="MaxEIG", bootstrap_samples=args.bootstrap_samples)

    output = args.output_dir.resolve()
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "episodes.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    pd.DataFrame(paired).to_csv(output / "paired_vs_maxeig.csv", index=False)
    plot_metric(summaries, "weighted_rmse", "Weighted RMSE (lower is better)", figures / "weighted_rmse.png", True)
    plot_metric(summaries, "return", "Mean episode return (higher is better)", figures / "return.png")
    plot_metric(summaries, "success", "Success rate (higher is better)", figures / "success_rate.png")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "episodes_per_checkpoint": args.episodes,
        "base_episode_seed": args.base_seed,
        "bootstrap_samples": args.bootstrap_samples,
        "domains": list(HELDOUT_DOMAINS),
        "primary_endpoint": "weighted_rmse (lower is better)",
        "primary_endpoint_status": "exploratory re-analysis; not preregistered",
        "checkpoint_sha256": {path.name: sha256(path) for path in checkpoints},
        "notes": [
            "Every method receives the same candidate episode seeds; answer randomness is seeded per episode.",
            "The trained policies have three checkpoint/training-seed replicates each.",
            "The hierarchical bootstrap resamples checkpoint seeds and episodes for method summaries.",
            "Paired difference intervals average available checkpoint seeds per episode and therefore condition on those trained checkpoints.",
            "The environment and outcomes are synthetic; results do not establish real-world hiring validity or fairness.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} episode rows and {len(summaries)} metric summaries to {output}")
    print(pd.DataFrame(summaries).query("metric in ['weighted_rmse', 'success', 'return']").to_string(index=False))


if __name__ == "__main__":
    main()
