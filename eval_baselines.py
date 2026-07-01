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
    least_loaded_edge_policy,
    local_only_policy,
    nearest_edge_policy,
    random_policy,
)


PolicyFn = Callable[[dict], list[int]]
PolicyBuilder = Callable[[MECEnv], PolicyFn]


POLICY_BUILDERS: dict[str, PolicyBuilder] = {
    "random": lambda env: build_random_policy(env),
    "local_only": lambda env: local_only_policy,
    "best_uplink": lambda env: best_uplink_rate_policy,
    "largest_queue": lambda env: largest_queue_policy,
    "nearest_edge": lambda env: nearest_edge_policy,
    "least_loaded_edge": lambda env: least_loaded_edge_policy,
}

METRIC_KEYS = [
    "total_reward",
    "completed_tasks",
    "dropped_tasks",
    "deadline_violations",
    "deadline_violation_rate",
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
    total_deadline_violations = 0
    total_delay = 0.0
    total_queue = 0.0
    steps = 0

    while True:
        action = policy(obs)

        obs, reward, done, info = env.step(action)
        total_reward += reward
        total_completed += int(info["completed_tasks"])
        total_dropped += int(info["dropped_tasks"])
        total_deadline_violations += int(info.get("deadline_violations", info["dropped_tasks"]))
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
        "deadline_violations": total_deadline_violations,
        "deadline_violation_rate": total_deadline_violations
        / max(total_completed + total_deadline_violations, 1),
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
        f"{metrics['deadline_violation_rate']:>14.3f}"
        f"{metrics['avg_delay']:>14.3f}"
        f"{metrics['avg_total_queue']:>18.3f}"
    )


def selected_policy_names(policy_arg: str) -> list[str]:
    if policy_arg == "all":
        return list(POLICY_BUILDERS)
    return [policy_arg]


def parse_float_tuple(value: str | None) -> tuple[float, ...] | None:
    if value is None or not value.strip():
        return None
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def parse_int_tuple(value: str | None) -> tuple[int, ...] | None:
    if value is None or not value.strip():
        return None
    return tuple(int(float(item.strip())) for item in value.split(",") if item.strip())


def parse_str_tuple(value: str | None) -> tuple[str, ...] | None:
    if value is None or not value.strip():
        return None
    return tuple(item.strip() for item in value.split(",") if item.strip())


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
        help="Optional path for aggregate metrics, for example experiment_records/baselines.csv.",
    )
    parser.add_argument(
        "--output-format",
        choices=["auto", "csv", "jsonl"],
        default="auto",
        help="Result file format. With auto, .jsonl writes JSONL and every other suffix writes CSV.",
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=None,
        help="Optional normalized task trace CSV. When set, task arrivals come from the trace.",
    )
    parser.add_argument(
        "--reward-preset",
        choices=["debug", "sla"],
        default="debug",
        help="Reward preset. debug preserves old experiments; sla emphasizes completion and deadline safety.",
    )
    parser.add_argument("--num-edge-servers", type=int, default=1, help="Number of edge/MEC servers.")
    parser.add_argument(
        "--edge-server-positions",
        type=str,
        default=None,
        help="Comma-separated server positions, for example 0,50,100.",
    )
    parser.add_argument(
        "--edge-server-compute-rates",
        type=str,
        default=None,
        help="Comma-separated server compute rates, for example 10,14,18.",
    )
    parser.add_argument(
        "--edge-server-coverage-radius",
        type=float,
        default=None,
        help="Optional coverage radius for every edge server.",
    )
    parser.add_argument(
        "--edge-selection-policy",
        choices=["nearest", "least_loaded"],
        default="nearest",
        help="Default target-server selection policy for legacy user-id-only actions.",
    )
    parser.add_argument("--task-type-count", type=int, default=1, help="Number of task classes.")
    parser.add_argument(
        "--task-type-names",
        type=str,
        default=None,
        help="Comma-separated task class names, for example light,normal,urgent.",
    )
    parser.add_argument(
        "--task-type-probabilities",
        type=str,
        default=None,
        help="Comma-separated task class sampling weights.",
    )
    parser.add_argument(
        "--task-cycles-per-unit-by-type",
        type=str,
        default=None,
        help="Comma-separated cycles-per-unit profile per task class.",
    )
    parser.add_argument(
        "--task-deadlines-by-type",
        type=str,
        default=None,
        help="Comma-separated deadline profile per task class.",
    )
    parser.add_argument(
        "--task-output-ratios-by-type",
        type=str,
        default=None,
        help="Comma-separated output-size/input-size ratios per task class.",
    )
    parser.add_argument(
        "--task-priorities-by-type",
        type=str,
        default=None,
        help="Comma-separated priority values per task class.",
    )
    args = parser.parse_args()

    policy_names = selected_policy_names(args.policy)
    print(
        f"episodes={args.episodes} seed={args.seed} policy={args.policy} "
        f"reward_preset={args.reward_preset} num_edge_servers={args.num_edge_servers} "
        f"task_type_count={args.task_type_count}"
    )
    print(
        f"{'policy':<18}"
        f"{'avg_reward':>14}"
        f"{'completed':>14}"
        f"{'dropped':>14}"
        f"{'ddl_rate':>14}"
        f"{'avg_delay':>14}"
        f"{'avg_total_queue':>18}"
    )
    print("-" * 110)

    aggregate_rows = []
    for index, name in enumerate(policy_names):
        policy_builder = POLICY_BUILDERS[name]
        env = MECEnv(
            MECConfig(
                random_seed=args.seed + index,
                task_trace_path=str(args.trace) if args.trace is not None else None,
                reward_preset=args.reward_preset,
                num_edge_servers=args.num_edge_servers,
                edge_server_positions=parse_float_tuple(args.edge_server_positions),
                edge_server_compute_rates=parse_float_tuple(args.edge_server_compute_rates),
                edge_server_coverage_radius=args.edge_server_coverage_radius,
                edge_selection_policy=args.edge_selection_policy,
                task_type_count=args.task_type_count,
                task_type_names=parse_str_tuple(args.task_type_names),
                task_type_probabilities=parse_float_tuple(args.task_type_probabilities),
                task_cycles_per_unit_by_type=parse_float_tuple(args.task_cycles_per_unit_by_type),
                task_deadlines_by_type=parse_int_tuple(args.task_deadlines_by_type),
                task_output_ratios_by_type=parse_float_tuple(args.task_output_ratios_by_type),
                task_priorities_by_type=parse_float_tuple(args.task_priorities_by_type),
            )
        )
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
