from __future__ import annotations

import argparse
import csv
import glob
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim import (
    MECEnv,
    SCENARIO_NAMES,
    best_uplink_rate_policy,
    build_runtime_config,
    flatten_observation,
    observation_vector_length,
)


DEFAULT_SCENARIOS = ("multi_edge_network_sla", "cloud_edge_sla")
REQUIRED_TRACE_COLUMNS = {
    "step",
    "user_id",
    "size",
    "cycles",
    "deadline",
    "upload",
    "task_type_id",
    "task_type",
    "output_size",
    "priority",
}
REQUIRED_INFO_KEYS = {
    "completed_tasks",
    "dropped_tasks",
    "deadline_violations",
    "avg_delay",
    "total_queue",
    "avg_edge_utilization",
    "network_data",
    "cloud_usage_ratio",
    "energy_used",
    "cloud_cost",
}


class Preflight:
    def __init__(self) -> None:
        self.checks: list[dict] = []

    def ok(self, name: str, detail: str, **extra) -> None:
        self.checks.append({"name": name, "status": "ok", "detail": detail, **extra})
        print(f"ok {name}: {detail}")

    def fail(self, name: str, detail: str, **extra) -> None:
        self.checks.append({"name": name, "status": "failed", "detail": detail, **extra})
        print(f"failed {name}: {detail}")

    def skip(self, name: str, detail: str, **extra) -> None:
        self.checks.append({"name": name, "status": "skipped", "detail": detail, **extra})
        print(f"skipped {name}: {detail}")

    @property
    def failed(self) -> list[dict]:
        return [check for check in self.checks if check["status"] == "failed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight checks before formal MEC DRL experiments.")
    parser.add_argument(
        "--trace",
        type=Path,
        default=Path("experiment_records/trace_profiles/alibaba_light_normal_urgent.csv"),
    )
    parser.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS), choices=SCENARIO_NAMES)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiment_records/preflight/iteration37"),
    )
    parser.add_argument(
        "--commands-file",
        type=Path,
        default=Path("experiment_records/colab_commands/formal_drl_commands_50k.txt"),
    )
    parser.add_argument(
        "--baseline-glob",
        default="experiment_records/formal_baselines/heuristic_trace_profile_e10/*/*.csv",
    )
    parser.add_argument(
        "--ppo-glob",
        default="experiment_records/ppo_mec/smoke_matrix_iteration35/*/*/metrics.jsonl",
    )
    return parser.parse_args()


def read_trace_header_and_count(path: Path) -> tuple[set[str], int]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = sum(1 for _ in reader)
        return set(reader.fieldnames or []), rows


def check_trace(preflight: Preflight, trace: Path) -> None:
    if not trace.exists():
        preflight.fail("trace_profile", f"trace 文件不存在: {trace}")
        return
    columns, row_count = read_trace_header_and_count(trace)
    missing = sorted(REQUIRED_TRACE_COLUMNS - columns)
    if missing:
        preflight.fail("trace_profile", f"trace 缺少列: {missing}", rows=row_count)
        return
    preflight.ok("trace_profile", f"trace 可读，rows={row_count}", rows=row_count)


def check_scenario_rollout(preflight: Preflight, scenario: str, trace: Path, seed: int) -> None:
    try:
        config = build_runtime_config(scenario=scenario, trace_path=trace, reward_preset="sla", seed=seed)
        env = MECEnv(config)
        obs, _ = env.reset(seed=seed)
        expected_len = observation_vector_length(config.num_users)
        actual_len = len(flatten_observation(obs))
        if actual_len != expected_len:
            preflight.fail(
                f"scenario_rollout:{scenario}",
                f"观测维度异常 actual={actual_len} expected={expected_len}",
            )
            return
        total_completed = 0
        total_violations = 0
        last_info = {}
        for _ in range(8):
            obs, _, done, last_info = env.step(best_uplink_rate_policy(obs))
            total_completed += int(last_info["completed_tasks"])
            total_violations += int(last_info.get("deadline_violations", 0))
            if done:
                break
        missing = sorted(REQUIRED_INFO_KEYS - set(last_info))
        if missing:
            preflight.fail(f"scenario_rollout:{scenario}", f"info 缺少指标: {missing}")
            return
        detail = (
            f"观测维度={actual_len}, completed={total_completed}, "
            f"deadline_violations={total_violations}"
        )
        preflight.ok(f"scenario_rollout:{scenario}", detail)
    except Exception as exc:  # noqa: BLE001
        preflight.fail(f"scenario_rollout:{scenario}", repr(exc))


def check_script_help(preflight: Preflight, python_exe: str, script: str) -> None:
    result = subprocess.run(
        [python_exe, script, "--help"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        preflight.ok(f"script_help:{script}", "命令行参数可解析")
    else:
        preflight.fail(
            f"script_help:{script}",
            "命令行参数解析失败",
            stderr_tail=result.stderr.strip().splitlines()[-6:],
        )


def check_command_file(preflight: Preflight, path: Path) -> None:
    if not path.exists():
        preflight.fail("colab_command_file", f"命令清单不存在: {path}")
        return
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    commands = [line for line in lines if line.startswith("python ")]
    algorithms = {
        "sb3_ppo": sum("run_sb3_ppo_mec.py" in line for line in commands),
        "sac": sum("run_sac_mec.py" in line for line in commands),
        "dreamerv3": sum("run_dreamer_mec.py" in line for line in commands),
    }
    if len(commands) != 30 or any(count != 10 for count in algorithms.values()):
        preflight.fail("colab_command_file", f"命令数量异常 total={len(commands)} alg={algorithms}")
        return
    preflight.ok("colab_command_file", f"命令数量正确 total={len(commands)} alg={algorithms}")


def run_tiny_baseline(preflight: Preflight, args: argparse.Namespace, scenario: str) -> Path:
    output = args.output_root / "tiny_baseline" / f"{scenario}_best_uplink_seed{args.seed}.csv"
    result = subprocess.run(
        [
            args.python,
            "eval_baselines.py",
            "--scenario",
            scenario,
            "--trace",
            str(args.trace),
            "--episodes",
            "1",
            "--seed",
            str(args.seed),
            "--policy",
            "best_uplink",
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0 and output.exists():
        preflight.ok(f"tiny_baseline:{scenario}", f"1 episode 输出正常: {output}")
    else:
        preflight.fail(
            f"tiny_baseline:{scenario}",
            "1 episode baseline 失败",
            stderr_tail=result.stderr.strip().splitlines()[-8:],
        )
    return output


def run_aggregate_check(preflight: Preflight, args: argparse.Namespace) -> None:
    baseline_matches = glob.glob(str(PROJECT_ROOT / args.baseline_glob))
    ppo_matches = glob.glob(str(PROJECT_ROOT / args.ppo_glob))
    if not baseline_matches:
        preflight.fail("aggregate_inputs", f"找不到 baseline 输入: {args.baseline_glob}")
        return
    if not ppo_matches:
        preflight.skip("aggregate_inputs", f"找不到 PPO smoke 输入: {args.ppo_glob}")
        return
    output = args.output_root / "aggregate_check.csv"
    summary = args.output_root / "aggregate_check_summary.csv"
    result = subprocess.run(
        [
            args.python,
            "scripts/aggregate_experiment_results.py",
            "--inputs",
            args.baseline_glob,
            args.ppo_glob,
            "--output",
            str(output),
            "--summary-output",
            str(summary),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0 and output.exists() and summary.exists():
        preflight.ok(
            "aggregate_check",
            f"聚合链路正常 baseline_files={len(baseline_matches)} ppo_files={len(ppo_matches)}",
        )
    else:
        preflight.fail(
            "aggregate_check",
            "聚合链路失败",
            stderr_tail=result.stderr.strip().splitlines()[-8:],
        )


def write_manifest(preflight: Preflight, args: argparse.Namespace) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "trace": str(args.trace),
        "scenarios": list(args.scenarios),
        "seed": args.seed,
        "python": args.python,
        "checks": preflight.checks,
        "failed_count": len(preflight.failed),
    }
    path = args.output_root / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest={path}")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    preflight = Preflight()

    check_trace(preflight, args.trace)
    for scenario in args.scenarios:
        check_scenario_rollout(preflight, scenario, args.trace, args.seed)
    for script in (
        "eval_baselines.py",
        "scripts/run_ppo_mec.py",
        "scripts/run_sb3_ppo_mec.py",
        "scripts/run_sac_mec.py",
        "scripts/run_dreamer_mec.py",
        "scripts/aggregate_experiment_results.py",
    ):
        check_script_help(preflight, args.python, script)
    check_command_file(preflight, args.commands_file)
    for scenario in args.scenarios:
        run_tiny_baseline(preflight, args, scenario)
    run_aggregate_check(preflight, args)
    write_manifest(preflight, args)
    if preflight.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
