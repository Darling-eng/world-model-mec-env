from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable
from pathlib import Path

from sim import (
    MECConfig,
    MECEnv,
    best_uplink_rate_policy,
    largest_queue_policy,
    local_only_policy,
    random_policy,
)


PolicyFn = Callable[[dict], list[int]]
PolicyBuilder = Callable[[MECEnv], PolicyFn]


POLICY_BUILDERS: dict[str, PolicyBuilder] = {
    "random": lambda env: build_random_policy(env),
    "local_only": lambda env: local_only_policy,
    "best_uplink": lambda env: best_uplink_rate_policy,
    "largest_queue": lambda env: largest_queue_policy,
}

METRIC_KEYS = [
    "total_reward",
    "completed_tasks",
    "dropped_tasks",
    "avg_delay",
    "avg_total_queue",
    "steps",
]


def build_random_policy(env: MECEnv) -> PolicyFn:
    def _policy(obs: dict) -> list[int]:
        return random_policy(obs, env.rng)

    return _policy


def run_episode(env: MECEnv, policy_name: str, policy: PolicyFn, seed: int) -> dict:
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    total_completed = 0
    total_dropped = 0
    total_delay = 0.0
    total_queue = 0.0
    steps = 0

    while True:
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
        "policy": policy_name,
        "seed": seed,
        "total_reward": total_reward,
        "completed_tasks": total_completed,
        "dropped_tasks": total_dropped,
        "avg_delay": total_delay / max(steps, 1),
        "avg_total_queue": total_queue / max(steps, 1),
        "steps": steps,
    }


def aggregate_results(results: list[dict]) -> dict:
    count = max(len(results), 1)
    metrics = {key: sum(item[key] for item in results) / count for key in METRIC_KEYS}
    if results:
        metrics["policy"] = results[0]["policy"]
        metrics["episodes"] = len(results)
    return metrics


def format_row(name: str, metrics: dict) -> str:
    return (
        f"{name:<18}"
        f"{metrics['total_reward']:>14.3f}"
        f"{metrics['completed_tasks']:>14.2f}"
        f"{metrics['dropped_tasks']:>14.2f}"
        f"{metrics['avg_delay']:>14.3f}"
        f"{metrics['avg_total_queue']:>18.3f}"
    )


def selected_policy_names(policy_arg: str) -> list[str]:
    if policy_arg == "all":
        return list(POLICY_BUILDERS)
    return [policy_arg]


def infer_output_format(path: Path, explicit_format: str) -> str:
    if explicit_format != "auto":
        return explicit_format
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return "jsonl"
    return "csv"


def write_results(rows: list[dict], output_path: Path, output_format: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "jsonl":
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return

    fieldnames = ["policy", "episodes", *METRIC_KEYS]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate heuristic baselines on the MEC environment.")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes.")
    parser.add_argument("--seed", type=int, default=7, help="Base random seed.")
    parser.add_argument(
        "--policy",
        choices=["all", *POLICY_BUILDERS.keys()],
        default="all",
        help="Policy to evaluate. Use 'all' to run every heuristic baseline.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for aggregate metrics, for example outputs/baselines.csv.",
    )
    parser.add_argument(
        "--output-format",
        choices=["auto", "csv", "jsonl"],
        default="auto",
        help="Result file format. With auto, .jsonl writes JSONL and every other suffix writes CSV.",
    )
    args = parser.parse_args()

    policy_names = selected_policy_names(args.policy)
    print(f"episodes={args.episodes} seed={args.seed} policy={args.policy}")
    print(
        f"{'policy':<18}"
        f"{'avg_reward':>14}"
        f"{'completed':>14}"
        f"{'dropped':>14}"
        f"{'avg_delay':>14}"
        f"{'avg_total_queue':>18}"
    )
    print("-" * 96)

    aggregate_rows = []
    for index, name in enumerate(policy_names):
        policy_builder = POLICY_BUILDERS[name]
        env = MECEnv(MECConfig(random_seed=args.seed + index))
        policy = policy_builder(env)
        episode_results = [
            run_episode(env, name, policy, seed=args.seed + index * 1000 + episode)
            for episode in range(args.episodes)
        ]
        metrics = aggregate_results(episode_results)
        aggregate_rows.append(metrics)
        print(format_row(name, metrics))

    if args.output is not None:
        output_format = infer_output_format(args.output, args.output_format)
        write_results(aggregate_rows, args.output, output_format)
        print(f"wrote_results={args.output} format={output_format}")


if __name__ == "__main__":
    main()
