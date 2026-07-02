from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim import SCENARIO_NAMES


DEFAULT_SCENARIOS = ("multi_edge_network_sla", "cloud_edge_sla")


def run_ppo(
    *,
    python_exe: str,
    scenario: str,
    seed: int,
    trace: Path,
    steps: int,
    rollout_steps: int,
    eval_episodes: int,
    log_dir: Path,
) -> subprocess.CompletedProcess:
    command = [
        python_exe,
        "scripts/run_ppo_mec.py",
        "--steps",
        str(steps),
        "--rollout-steps",
        str(rollout_steps),
        "--eval-episodes",
        str(eval_episodes),
        "--seed",
        str(seed),
        "--scenario",
        scenario,
        "--trace",
        str(trace),
        "--reward-preset",
        "sla",
        "--log-dir",
        str(log_dir),
    ]
    return subprocess.run(command, text=True, capture_output=True, check=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local lightweight PPO smoke matrix for MEC scenarios.")
    parser.add_argument(
        "--trace",
        type=Path,
        default=Path("experiment_records/trace_profiles/alibaba_light_normal_urgent.csv"),
    )
    parser.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS), choices=SCENARIO_NAMES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27])
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--output-root", type=Path, default=Path("experiment_records/ppo_mec/smoke_matrix"))
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.trace.exists():
        raise FileNotFoundError(args.trace)
    args.output_root.mkdir(parents=True, exist_ok=True)
    runs = []
    failures = []
    for scenario in args.scenarios:
        for seed in args.seeds:
            log_dir = args.output_root / scenario / f"seed{seed}"
            result = run_ppo(
                python_exe=args.python,
                scenario=scenario,
                seed=seed,
                trace=args.trace,
                steps=args.steps,
                rollout_steps=args.rollout_steps,
                eval_episodes=args.eval_episodes,
                log_dir=log_dir,
            )
            metrics_path = log_dir / "metrics.jsonl"
            record = {
                "scenario": scenario,
                "seed": seed,
                "steps": args.steps,
                "rollout_steps": args.rollout_steps,
                "eval_episodes": args.eval_episodes,
                "trace": str(args.trace),
                "log_dir": str(log_dir),
                "metrics": str(metrics_path),
                "returncode": result.returncode,
                "stdout_tail": result.stdout.strip().splitlines()[-8:],
                "stderr_tail": result.stderr.strip().splitlines()[-8:],
                "metrics_exists": metrics_path.exists(),
            }
            runs.append(record)
            status = "ok" if result.returncode == 0 and metrics_path.exists() else "failed"
            print(f"{status} scenario={scenario} seed={seed} metrics={metrics_path}")
            if status != "ok":
                failures.append(record)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "trace": str(args.trace),
        "scenarios": list(args.scenarios),
        "seeds": list(args.seeds),
        "steps": args.steps,
        "rollout_steps": args.rollout_steps,
        "eval_episodes": args.eval_episodes,
        "run_count": len(runs),
        "failure_count": len(failures),
        "runs": runs,
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest={manifest_path}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
