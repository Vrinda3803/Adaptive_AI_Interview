"""Tabular Q-learning policy for choosing the assessment category.

The trained PPO checkpoint chooses skill, difficulty, and conceptual/coding type.
This second RL policy chooses among context-legal content categories. It never
generates wording; Gemini and curated question banks only realize its choice.
"""

import json
from pathlib import Path

import numpy as np


GENERAL_CATEGORIES = (
    "resume_based",
    "skill_based",
    "project_based",
    "theoretical",
    "scenario_based",
)
ALL_CATEGORIES = GENERAL_CATEGORIES + ("follow_up", "hr_behavioral", "coding")
CATEGORY_BIT = {name: 1 << index for index, name in enumerate(GENERAL_CATEGORIES)}
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "category_q_policy.json"


def legal_categories(skill, question_type, state):
    """Return the category action mask allowed by the selected PPO action/context."""
    if int(question_type) == 1:
        return ["coding"]
    if skill == "Communication & Leadership":
        return ["hr_behavioral"]

    options = ["skill_based", "theoretical", "scenario_based"]
    if str(state.get("resume_text", "")).strip():
        options.append("resume_based")
    if str(state.get("projects", "")).strip():
        options.append("project_based")

    history = state.get("history", [])
    followup_count = len(state.get("followup_skills", []))
    last = history[-1] if history else None
    if (
        last
        and last.get("skill") == skill
        and state.get("last_answer")
        and followup_count < 2
    ):
        options.append("follow_up")
    return options


def encode_state(history, has_resume, has_projects, followup_available, last_score, remaining_fraction):
    """Compact observable category-policy state; no hidden candidate mastery."""
    covered = 0
    for item in history:
        category = item.get("category", "") if isinstance(item, dict) else str(item)
        covered |= CATEGORY_BIT.get(category, 0)
    answer_bin = 0 if last_score is None else 1 if last_score < 0.45 else 2 if last_score < 0.75 else 3
    budget_bin = int(np.clip(float(remaining_fraction) * 4, 0, 3))
    value = covered
    value = value * 2 + int(bool(has_resume))
    value = value * 2 + int(bool(has_projects))
    value = value * 2 + int(bool(followup_available))
    value = value * 4 + answer_bin
    return value * 4 + budget_bin


def train_category_q_policy(episodes=50000, seed=20261103, alpha=0.12, gamma=0.92):
    """Train a contextual category selector in a synthetic coverage/relevance task."""
    rng = np.random.default_rng(seed)
    q = {}
    categories = GENERAL_CATEGORIES + ("follow_up",)

    def row(state_id):
        return q.setdefault(str(state_id), {name: 0.0 for name in categories})

    for episode in range(int(episodes)):
        has_resume = bool(rng.integers(0, 2))
        has_projects = bool(rng.integers(0, 2))
        max_turns = int(rng.integers(6, 11))
        coverage = []
        followups = 0
        score = None
        epsilon = max(0.025, 0.35 * (1.0 - episode / max(episodes, 1)))

        for turn in range(max_turns):
            followup_available = turn > 0 and followups < 2 and bool(rng.random() < 0.35)
            options = ["skill_based", "theoretical", "scenario_based"]
            if has_resume:
                options.append("resume_based")
            if has_projects:
                options.append("project_based")
            if followup_available:
                options.append("follow_up")
            remaining = (max_turns - turn) / max_turns
            state_id = encode_state(coverage, has_resume, has_projects, followup_available, score, remaining)
            values = row(state_id)
            if rng.random() < epsilon:
                action = str(rng.choice(options))
            else:
                best = max(values.get(name, 0.0) for name in options)
                tied = [name for name in options if values.get(name, 0.0) == best]
                action = str(rng.choice(tied))

            newly_covered = action in GENERAL_CATEGORIES and action not in coverage
            reward = 1.0 if newly_covered else -0.18 if action in coverage else 0.05
            if action == "resume_based" and has_resume:
                reward += 0.18
            if action == "project_based" and has_projects:
                reward += 0.18
            if action == "follow_up":
                reward += 0.12 if score is not None and score < 0.75 else -0.03
                followups += 1

            if newly_covered:
                coverage.append(action)
            next_score = float(rng.beta(2.0, 2.0))
            done = turn == max_turns - 1 or len(coverage) == len(options) and not followup_available
            if done:
                future = 0.0
            else:
                next_followup = followups < 2 and bool(rng.random() < 0.35)
                next_options = ["skill_based", "theoretical", "scenario_based"]
                if has_resume:
                    next_options.append("resume_based")
                if has_projects:
                    next_options.append("project_based")
                if next_followup:
                    next_options.append("follow_up")
                next_id = encode_state(coverage, has_resume, has_projects, next_followup, next_score, (max_turns - turn - 1) / max_turns)
                next_values = row(next_id)
                future = max(next_values.get(name, 0.0) for name in next_options)
            values[action] += alpha * (reward + gamma * future - values[action])
            score = next_score

    return {
        "format_version": 1,
        "algorithm": "tabular_q_learning",
        "seed": int(seed),
        "training_episodes": int(episodes),
        "state_action_values": q,
    }


class CategoryRLAgent:
    def __init__(self, model_path=DEFAULT_MODEL):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Category RL policy not found: {model_path}. "
                "Run `python train_category_policy.py` to create it."
            )
        self.model_path = model_path
        self.model = json.loads(model_path.read_text(encoding="utf-8"))
        if self.model.get("algorithm") != "tabular_q_learning":
            raise ValueError(f"Unsupported category policy: {self.model.get('algorithm')}")
        self.q_values = self.model["state_action_values"]

    def decide(self, skill, question_type, state, excluded_categories=()):
        all_options = legal_categories(skill, question_type, state)
        excluded = set(excluded_categories or ())
        options = [name for name in all_options if name not in excluded]
        history = state.get("history", [])
        followup_count = len(state.get("followup_skills", []))
        last = history[-1] if history else None
        followup_available = "follow_up" in all_options
        state_id = encode_state(
            history,
            bool(str(state.get("resume_text", "")).strip()),
            bool(str(state.get("projects", "")).strip()),
            followup_available,
            state.get("last_score"),
            state.get("remaining_fraction", 1.0),
        )
        learned = self.q_values.get(str(state_id), {})
        # Fixed tie order makes online inference deterministic and reproducible.
        category = max(options, key=lambda name: learned.get(name, 0.0)) if options else None
        return {
            "category": category,
            "legal_categories": options,
            "excluded_categories": sorted(excluded),
            "q_values": {name: float(learned.get(name, 0.0)) for name in options},
            "state_id": state_id,
            "policy": "category_q_policy",
        }
