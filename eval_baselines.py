from __future__ import annotations

import argparse
from collections.abc import Callable

from sim import (
    MECConfig,
    MECEnv,
    best_uplink_rate_policy,
    largest_queue_policy,
    local_only_policy,
    random_policy,
)


PolicyFn = Callable[[dict], list[int]]


def run_episode(env: MECEnv, policy_name: str, policy: PolicyFn, seed: int) -> dict:
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    total_completed = 0
    total_dropped = 0
    total_delay = 0.0
    total_queue = 0.0
    steps = 0

    while True:
        if policy_name == "random":
            action = random_policy(obs, env.rng)
        else:
            action = policy(obs)

        obs, reward, done, info = env.step(action)
        total_reward += reward
        total_completed += int(info["completed_tasks"])
        total_dropped += int(info["dropped_tasks"])
        total_delay += float(info["avg_delay"])
        total_queue += float(info["total_queue"])
        steps += 1
        if done:
            break

    return {
        "total_reward": total_reward,
        "completed_tasks": total_completed,
        "dropped_tasks": total_dropped,
        "average_delay": total_delay / max(steps, 1),
        "average_queue_length": total_queue / max(steps, 1),
        "steps": steps,
    }


def aggregate_results(results: list[dict]) -> dict:
    count = max(len(results), 1)
    keys = [
        "total_reward",
        "completed_tasks",
        "dropped_tasks",
        "average_delay",
        "average_queue_length",
        "steps",
    ]
    return {key: sum(item[key] for item in results) / count for key in keys}


def format_row(name: str, metrics: dict) -> str:
    return (
        f"{name:<18}"
        f"{metrics['total_reward']:>14.3f}"
        f"{metrics['completed_tasks']:>14.2f}"
        f"{metrics['dropped_tasks']:>14.2f}"
        f"{metrics['average_delay']:>14.3f}"
        f"{metrics['average_queue_length']:>16.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate heuristic baselines on the MEC environment.")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes.")
    parser.add_argument("--seed", type=int, default=7, help="Base random seed.")
    args = parser.parse_args()

    policies: dict[str, PolicyFn] = {
        "random": local_only_policy,
        "local_only": local_only_policy,
        "best_uplink": best_uplink_rate_policy,
        "largest_queue": largest_queue_policy,
    }

    print(f"episodes={args.episodes} seed={args.seed}")
    print(
        f"{'policy':<18}"
        f"{'avg_reward':>14}"
        f"{'completed':>14}"
        f"{'dropped':>14}"
        f"{'avg_delay':>14}"
        f"{'avg_queue':>16}"
    )
    print("-" * 90)

    for index, (name, policy) in enumerate(policies.items()):
        env = MECEnv(MECConfig(random_seed=args.seed + index))
        episode_results = [
            run_episode(env, name, policy, seed=args.seed + index * 1000 + episode)
            for episode in range(args.episodes)
        ]
        metrics = aggregate_results(episode_results)
        print(format_row(name, metrics))


if __name__ == "__main__":
    main()
