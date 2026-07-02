from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim import SCENARIO_NAMES


DEFAULT_POLICIES = ("local_only", "nearest_edge", "least_loaded_edge", "cloud_edge")


def parse_csv_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_baseline(
    *,
    python_exe: str,
    scenario: str,
    policy: str,
    seed: int,
    episodes: int,
    output_path: Path,
) -> subprocess.CompletedProcess:
    command = [
        python_exe,
        "eval_baselines.py",
        "--scenario",
        scenario,
        "--episodes",
        str(episodes),
        "--seed",
        str(seed),
        "--policy",
        policy,
        "--output",
        str(output_path),
    ]
    return subprocess.run(command, text=True, capture_output=True, check=False)


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a fixed scenario-regression matrix for MEC baseline sanity checks."
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=list(SCENARIO_NAMES),
        choices=SCENARIO_NAMES,
        help="Named scenarios to run.",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=list(DEFAULT_POLICIES),
        help="Baseline policies to run for each scenario.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[7], help="Seeds for each scenario/policy pair.")
    parser.add_argument("--episodes", type=int, default=2, help="Episodes per run.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiment_records/scenario_regression/iteration29"),
        help="Root folder for scenario-regression CSV files and manifest.json.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch eval_baselines.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    runs = []
    failures = []

    for scenario in args.scenarios:
        for seed in args.seeds:
            for policy in args.policies:
                output_path = args.output_root / scenario / f"{policy}_seed{seed}.csv"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                result = run_baseline(
                    python_exe=args.python,
                    scenario=scenario,
                    policy=policy,
                    seed=seed,
                    episodes=args.episodes,
                    output_path=output_path,
                )
                rows = parse_csv_rows(output_path) if output_path.exists() else []
                run_record = {
                    "scenario": scenario,
                    "policy": policy,
                    "seed": seed,
                    "episodes": args.episodes,
                    "output": str(output_path),
                    "returncode": result.returncode,
                    "stdout_tail": result.stdout.strip().splitlines()[-5:],
                    "stderr_tail": result.stderr.strip().splitlines()[-5:],
                    "rows": rows,
                }
                runs.append(run_record)
                status = "ok" if result.returncode == 0 else "failed"
                print(f"{status} scenario={scenario} policy={policy} seed={seed} output={output_path}")
                if result.returncode != 0:
                    failures.append(run_record)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(args.output_root),
        "episodes": args.episodes,
        "scenarios": list(args.scenarios),
        "policies": list(args.policies),
        "seeds": list(args.seeds),
        "run_count": len(runs),
        "failure_count": len(failures),
        "runs": runs,
    }
    manifest_path = args.output_root / "manifest.json"
    write_manifest(manifest_path, manifest)
    print(f"manifest={manifest_path}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
