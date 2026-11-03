"""Evaluation helpers for the live custom PyTorch policy on held-out domains."""

import numpy as np

from .environment import EnvCfg, InterviewEnv, HELDOUT_DOMAINS


def evaluate_heldout(policy, episodes=25, seed=20261103):
    return _evaluate(policy, EnvCfg(allowed_domains=HELDOUT_DOMAINS), episodes, seed)


def _evaluate(policy, cfg, episodes, seed):
    env = InterviewEnv(cfg)
    rows = []
    for offset in range(int(episodes)):
        obs = env.reset(seed=seed + offset)
        done = False
        total_return = 0.0
        invalid_actions = 0
        info = {}
        while not done:
            action = policy(obs, env)
            obs, reward, done, info = env.step(action)
            total_return += float(reward)
            invalid_actions += int(info.get("invalid", 0))
        final = info.get("final", {})
        rows.append({
            "episode": offset + 1,
            "return": total_return,
            "success": final.get("success", 0),
            "weighted_rmse": final.get("wrmse", np.nan),
            "decision_accuracy": final.get("dec_acc", np.nan),
            "coverage": final.get("coverage", np.nan),
            "questions": final.get("questions_asked", env.t),
            "invalid_actions": invalid_actions,
        })
    env.close()
    return rows


def evaluate_ablation(policy, episodes=10, seed=31000):
    variants = {
        "Full PPO": {},
        "Without Knowledge Graph": {"use_graph": False},
        "Without EIG features": {"use_eig_feat": False},
        "Without uncertainty features": {"use_sigma_feat": False},
        "Without action masking": {"use_mask": False},
        "Without resume prior": {"use_resume_prior": False},
    }
    rows = []
    for offset, (name, changes) in enumerate(variants.items()):
        results = _evaluate(policy, EnvCfg(**changes), episodes, seed + offset * episodes)
        rows.append({
            "variant": name,
            "episodes": len(results),
            "mean_return": float(np.mean([r["return"] for r in results])),
            "success_rate": float(np.mean([r["success"] for r in results])),
            "mean_weighted_rmse": _finite_mean([r["weighted_rmse"] for r in results]),
            "mean_invalid_actions": float(np.mean([r["invalid_actions"] for r in results])),
        })
    return rows


def evaluate_noise_robustness(policy, episodes=10, seed=41000):
    rows = []
    for offset, noise in enumerate((0.0, 0.05, 0.15, 0.25)):
        results = _evaluate(policy, EnvCfg(allowed_domains=HELDOUT_DOMAINS, eps_true=noise), episodes, seed + offset * episodes)
        rows.append({
            "true_evaluator_noise": noise,
            "episodes": len(results),
            "mean_return": float(np.mean([r["return"] for r in results])),
            "success_rate": float(np.mean([r["success"] for r in results])),
            "mean_weighted_rmse": _finite_mean([r["weighted_rmse"] for r in results]),
        })
    return rows


def _finite_mean(values):
    values = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(values)) if values else float("nan")
