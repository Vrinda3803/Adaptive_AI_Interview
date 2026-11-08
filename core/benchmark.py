from pathlib import Path
import numpy as np
from .action_net import load_policy
from .environment import EnvCfg, HELDOUT_DOMAINS, InterviewEnv

class HeuristicPolicy:
    def __init__(self, method, seed=0):
        self.method = method
        self.rng = np.random.default_rng(seed)
        self.name = method
        
    def __call__(self, obs, env=None):
        valid = np.flatnonzero(obs["m"])
        if not len(valid):
            raise RuntimeError("The environment returned no valid actions before termination.")
        if self.method == "Random":
            return int(self.rng.choice(valid))
        if self.method == "Uncertainty":
            scores = obs["f"][valid, 2]
        elif self.method == "MaxEIG":
            scores = obs["f"][valid, 8]
        else:
            raise ValueError(f"Unknown heuristic: {self.method}")
        return int(valid[int(np.argmax(scores))])


def _run_episode(policy, episode_seed, model_seed, method):
    env = InterviewEnv(EnvCfg(allowed_domains=HELDOUT_DOMAINS))
    obs = env.reset(seed=int(episode_seed))
    done, total_return, invalid = False, 0.0, 0
    info = {}
    while not done:
        action = int(policy(obs, env))
        obs, reward, done, info = env.step(action)
        total_return += float(reward)
        invalid += int(info.get("invalid", 0))
    final = info.get("final", {})
    return {
        "method": method,
        "model_seed": model_seed,
        "episode_seed": int(episode_seed),
        "return": total_return,
        "success": int(final.get("success", 0)),
        "weighted_rmse": float(final.get("wrmse", np.nan)),
        "decision_accuracy": float(final.get("dec_acc", np.nan)),
        "coverage": float(final.get("coverage", np.nan)),
        "questions": int(final.get("questions_asked", env.t)),
        "invalid_actions": invalid,
        "stopped_early": int(final.get("stopped_early", False)),
    }

def run_seeded_benchmark(checkpoint_dir, episodes=300, base_seed=91000):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob("*_seed*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No *_seed*.pt benchmark checkpoints in {checkpoint_dir}")
    episode_seeds = np.arange(int(base_seed), int(base_seed) + int(episodes), dtype=int)
    rows = []
    for checkpoint in checkpoints:
        policy = load_policy(checkpoint)
        method = policy.name
        model_seed = int(checkpoint.stem.rsplit("seed", 1)[1])
        for episode_seed in episode_seeds:
            rows.append(_run_episode(policy, episode_seed, model_seed, method))
    for method in ("Random", "Uncertainty", "MaxEIG"):
        for episode_seed in episode_seeds:
            policy = HeuristicPolicy(method, seed=int(episode_seed) + 1_000_003)
            rows.append(_run_episode(policy, episode_seed, "heuristic", method))
    return rows


def summarize_seeded_benchmark(rows, bootstrap_samples=10000, seed=20261103):
    rng = np.random.default_rng(seed)
    metrics = ("return", "success", "weighted_rmse", "decision_accuracy", "coverage", "questions", "invalid_actions")
    summaries = []
    methods = sorted({row["method"] for row in rows})
    for method in methods:
        method_rows = [r for r in rows if r["method"] == method]
        groups = []
        for model_seed in sorted({r["model_seed"] for r in method_rows}, key=str):
            group = [r for r in method_rows if r["model_seed"] == model_seed]
            groups.append({m: np.asarray([float(r[m]) for r in group], dtype=float) for m in metrics})
        for metric in metrics:
            point = float(np.mean([np.nanmean(g[metric]) for g in groups]))
            draws = np.empty(int(bootstrap_samples), dtype=float)
            for b in range(len(draws)):
                seed_ids = rng.integers(0, len(groups), size=len(groups)) if len(groups) > 1 else np.zeros(1, dtype=int)
                sampled_means = []
                for seed_id in seed_ids:
                    values = groups[int(seed_id)][metric]
                    values = values[np.isfinite(values)]
                    if values.size:
                        sampled_means.append(float(np.mean(rng.choice(values, size=values.size, replace=True))))
                draws[b] = np.mean(sampled_means) if sampled_means else np.nan
            lo, hi = np.nanquantile(draws, [0.025, 0.975])
            summaries.append({
                "method": method,
                "metric": metric,
                "estimate": point,
                "ci95_low": float(lo),
                "ci95_high": float(hi),
                "model_seed_replicates": len(groups) if method not in {"Random", "Uncertainty", "MaxEIG"} else 0,
                "episodes_per_seed": len(groups[0][metric]),
                "bootstrap_samples": int(bootstrap_samples),
            })
    return summaries


def paired_difference_intervals(rows, metric="weighted_rmse", reference="MaxEIG", bootstrap_samples=10000, seed=20261104):
    rng = np.random.default_rng(seed)
    methods = sorted({r["method"] for r in rows} - {reference})
    ref_by_episode = {r["episode_seed"]: float(r[metric]) for r in rows if r["method"] == reference}
    results = []
    for method in methods:
        by_episode = {}
        for r in rows:
            if r["method"] == method and r["episode_seed"] in ref_by_episode:
                by_episode.setdefault(r["episode_seed"], []).append(float(r[metric]))
        seeds = sorted(set(by_episode) & set(ref_by_episode))
        diff = np.asarray([np.mean(by_episode[s]) - ref_by_episode[s] for s in seeds], dtype=float)
        boot = np.asarray([np.mean(rng.choice(diff, size=len(diff), replace=True)) for _ in range(int(bootstrap_samples))])
        lo, hi = np.quantile(boot, [0.025, 0.975])
        results.append({
            "metric": metric,
            "comparison": f"{method} - {reference}",
            "estimate": float(diff.mean()),
            "paired_ci95_low": float(lo),
            "paired_ci95_high": float(hi),
            "paired_episode_seeds": len(diff),
            "interpretation": "negative favors the first method" if metric == "weighted_rmse" else "positive favors the first method",
        })
    return results
