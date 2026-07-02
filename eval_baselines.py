from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Callable
from pathlib import Path

from sim import (
    MECConfig,
    MECEnv,
    SCENARIO_NAMES,
    best_uplink_rate_policy,
    build_scenario_config,
    cloud_edge_policy,
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
    "cloud_edge": lambda env: cloud_edge_policy,
}

METRIC_KEYS = [
    "total_reward",
    "completed_tasks",
    "dropped_tasks",
    "deadline_violations",
    "deadline_violation_rate",
    "avg_delay",
    "avg_total_queue",
    "avg_edge_utilization",
    "avg_uplink_data",
    "avg_downlink_data",
    "avg_network_data",
    "avg_cloud_utilization",
    "avg_cloud_usage_ratio",
    "avg_energy_used",
    "avg_cloud_cost",
    "steps",
]

CONFIG_KEYS = [
    "seed",
    "scenario",
    "reward_preset",
    "num_edge_servers",
    "edge_selection_policy",
    "task_type_count",
    "enable_uplink_contention",
    "enable_downlink_transmission",
    "enable_cloud_fallback",
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
    total_edge_utilization = 0.0
    total_uplink_data = 0.0
    total_downlink_data = 0.0
    total_network_data = 0.0
    total_cloud_utilization = 0.0
    total_cloud_usage_ratio = 0.0
    total_energy_used = 0.0
    total_cloud_cost = 0.0
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
        total_edge_utilization += float(info.get("avg_edge_utilization", 0.0))
        total_uplink_data += float(info.get("uplink_data", 0.0))
        total_downlink_data += float(info.get("downlink_data", 0.0))
        total_network_data += float(info.get("network_data", 0.0))
        total_cloud_utilization += float(info.get("cloud_utilization", 0.0))
        total_cloud_usage_ratio += float(info.get("cloud_usage_ratio", 0.0))
        total_energy_used += float(info.get("energy_used", 0.0))
        total_cloud_cost += float(info.get("cloud_cost", 0.0))
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
        "avg_edge_utilization": total_edge_utilization / max(steps, 1),
        "avg_uplink_data": total_uplink_data / max(steps, 1),
        "avg_downlink_data": total_downlink_data / max(steps, 1),
        "avg_network_data": total_network_data / max(steps, 1),
        "avg_cloud_utilization": total_cloud_utilization / max(steps, 1),
        "avg_cloud_usage_ratio": total_cloud_usage_ratio / max(steps, 1),
        "avg_energy_used": total_energy_used / max(steps, 1),
        "avg_cloud_cost": total_cloud_cost / max(steps, 1),
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
        f"{metrics['avg_edge_utilization']:>14.3f}"
        f"{metrics['avg_network_data']:>16.3f}"
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


def scenario_metadata(args: argparse.Namespace, config: MECConfig) -> dict:
    return {
        "seed": config.random_seed,
        "scenario": args.scenario or "custom",
        "reward_preset": config.reward_preset,
        "num_edge_servers": config.num_edge_servers,
        "edge_selection_policy": config.edge_selection_policy,
        "task_type_count": config.task_type_count,
        "enable_uplink_contention": config.enable_uplink_contention,
        "enable_downlink_transmission": config.enable_downlink_transmission,
        "enable_cloud_fallback": config.enable_cloud_fallback,
    }


def write_results(rows: list[dict], output_path: Path, output_format: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "jsonl":
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return

    fieldnames = ["policy", "episodes", *CONFIG_KEYS, *METRIC_KEYS]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def arg_was_provided(name: str) -> bool:
    return name in sys.argv[1:]


def apply_cli_overrides(config: MECConfig, args: argparse.Namespace, *, random_seed: int) -> MECConfig:
    config.random_seed = random_seed
    if args.trace is not None:
        config.task_trace_path = str(args.trace)
    if arg_was_provided("--reward-preset"):
        config.reward_preset = args.reward_preset
    if arg_was_provided("--num-edge-servers"):
        config.num_edge_servers = args.num_edge_servers
    if args.edge_server_positions is not None:
        config.edge_server_positions = parse_float_tuple(args.edge_server_positions)
    if args.edge_server_compute_rates is not None:
        config.edge_server_compute_rates = parse_float_tuple(args.edge_server_compute_rates)
    if arg_was_provided("--edge-server-coverage-radius"):
        config.edge_server_coverage_radius = args.edge_server_coverage_radius
    if arg_was_provided("--edge-selection-policy"):
        config.edge_selection_policy = args.edge_selection_policy
    if arg_was_provided("--base-downlink-rate"):
        config.base_downlink_rate = args.base_downlink_rate
    if args.enable_uplink_contention:
        config.enable_uplink_contention = True
    if args.enable_downlink_transmission:
        config.enable_downlink_transmission = True
    if args.enable_cloud_fallback:
        config.enable_cloud_fallback = True
    if arg_was_provided("--cloud-compute-rate"):
        config.cloud_compute_rate = args.cloud_compute_rate
    if arg_was_provided("--cloud-wan-upload-rate"):
        config.cloud_wan_upload_rate = args.cloud_wan_upload_rate
    if arg_was_provided("--cloud-wan-downlink-rate"):
        config.cloud_wan_downlink_rate = args.cloud_wan_downlink_rate
    if arg_was_provided("--cloud-wan-delay-steps"):
        config.cloud_wan_delay_steps = args.cloud_wan_delay_steps
    if arg_was_provided("--task-type-count"):
        config.task_type_count = args.task_type_count
    if args.task_type_names is not None:
        config.task_type_names = parse_str_tuple(args.task_type_names)
    if args.task_type_probabilities is not None:
        config.task_type_probabilities = parse_float_tuple(args.task_type_probabilities)
    if args.task_cycles_per_unit_by_type is not None:
        config.task_cycles_per_unit_by_type = parse_float_tuple(args.task_cycles_per_unit_by_type)
    if args.task_deadlines_by_type is not None:
        config.task_deadlines_by_type = parse_int_tuple(args.task_deadlines_by_type)
    if args.task_output_ratios_by_type is not None:
        config.task_output_ratios_by_type = parse_float_tuple(args.task_output_ratios_by_type)
    if args.task_priorities_by_type is not None:
        config.task_priorities_by_type = parse_float_tuple(args.task_priorities_by_type)
    return config


def build_config_from_args(args: argparse.Namespace, *, random_seed: int) -> MECConfig:
    if args.scenario is not None:
        scenario_config = build_scenario_config(args.scenario, MECConfig(random_seed=random_seed))
        return apply_cli_overrides(scenario_config, args, random_seed=random_seed)
    return MECConfig(
        random_seed=random_seed,
        task_trace_path=str(args.trace) if args.trace is not None else None,
        reward_preset=args.reward_preset,
        num_edge_servers=args.num_edge_servers,
        edge_server_positions=parse_float_tuple(args.edge_server_positions),
        edge_server_compute_rates=parse_float_tuple(args.edge_server_compute_rates),
        edge_server_coverage_radius=args.edge_server_coverage_radius,
        edge_selection_policy=args.edge_selection_policy,
        base_downlink_rate=args.base_downlink_rate,
        enable_uplink_contention=args.enable_uplink_contention,
        enable_downlink_transmission=args.enable_downlink_transmission,
        enable_cloud_fallback=args.enable_cloud_fallback,
        cloud_compute_rate=args.cloud_compute_rate,
        cloud_wan_upload_rate=args.cloud_wan_upload_rate,
        cloud_wan_downlink_rate=args.cloud_wan_downlink_rate,
        cloud_wan_delay_steps=args.cloud_wan_delay_steps,
        task_type_count=args.task_type_count,
        task_type_names=parse_str_tuple(args.task_type_names),
        task_type_probabilities=parse_float_tuple(args.task_type_probabilities),
        task_cycles_per_unit_by_type=parse_float_tuple(args.task_cycles_per_unit_by_type),
        task_deadlines_by_type=parse_int_tuple(args.task_deadlines_by_type),
        task_output_ratios_by_type=parse_float_tuple(args.task_output_ratios_by_type),
        task_priorities_by_type=parse_float_tuple(args.task_priorities_by_type),
    )


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
    parser.add_argument(
        "--scenario",
        choices=SCENARIO_NAMES,
        default=None,
        help="Optional named scenario preset. Explicit CLI values override the preset where provided.",
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
    parser.add_argument(
        "--base-downlink-rate",
        type=float,
        default=16.0,
        help="Base downlink rate used when downlink transmission is enabled.",
    )
    parser.add_argument(
        "--enable-uplink-contention",
        action="store_true",
        help="Share each edge server's uplink capacity among simultaneous uploads.",
    )
    parser.add_argument(
        "--enable-downlink-transmission",
        action="store_true",
        help="Require computed edge tasks to return output data over a downlink stage before completion.",
    )
    parser.add_argument(
        "--enable-cloud-fallback",
        action="store_true",
        help="Allow explicit cloud fallback actions with target server id -1.",
    )
    parser.add_argument("--cloud-compute-rate", type=float, default=40.0, help="Cloud compute rate.")
    parser.add_argument("--cloud-wan-upload-rate", type=float, default=4.0, help="WAN upload rate to cloud.")
    parser.add_argument("--cloud-wan-downlink-rate", type=float, default=6.0, help="WAN downlink rate from cloud.")
    parser.add_argument("--cloud-wan-delay-steps", type=int, default=2, help="Fixed cloud WAN delay in steps.")
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
    display_config = build_config_from_args(args, random_seed=args.seed)
    print(
        f"episodes={args.episodes} seed={args.seed} policy={args.policy} "
        f"reward_preset={display_config.reward_preset} num_edge_servers={display_config.num_edge_servers} "
        f"task_type_count={display_config.task_type_count} scenario={args.scenario or 'custom'}"
    )
    print(
        f"{'policy':<18}"
        f"{'avg_reward':>14}"
        f"{'completed':>14}"
        f"{'dropped':>14}"
        f"{'ddl_rate':>14}"
        f"{'avg_delay':>14}"
        f"{'avg_total_queue':>18}"
        f"{'edge_util':>14}"
        f"{'net_data':>16}"
    )
    print("-" * 140)

    aggregate_rows = []
    for index, name in enumerate(policy_names):
        policy_builder = POLICY_BUILDERS[name]
        config = build_config_from_args(args, random_seed=args.seed + index)
        env = MECEnv(config)
        policy = policy_builder(env)
        episode_results = [
            run_episode(env, name, policy, seed=args.seed + index * 1000 + episode)
            for episode in range(args.episodes)
        ]
        metrics = aggregate_results(episode_results)
        metrics.update(scenario_metadata(args, config))
        aggregate_rows.append(metrics)
        print(format_row(name, metrics))

    if args.output is not None:
        output_format = infer_output_format(args.output, args.output_format)
        write_results(aggregate_rows, args.output, output_format)
        print(f"wrote_results={args.output} format={output_format}")


if __name__ == "__main__":
    main()
