"""Train the category-selection Q policy used with the PPO interviewer."""

import argparse
import json
from pathlib import Path

from core.category_agent import DEFAULT_MODEL, train_category_q_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    model = train_category_q_policy(episodes=args.episodes, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, indent=2), encoding="utf-8")
    print(f"Saved category Q policy trained for {args.episodes:,} episodes to {args.output.resolve()}")


if __name__ == "__main__":
    main()
