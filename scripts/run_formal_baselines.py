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


DEFAULT_SCENARIOS = ("multi_edge_network_sla", "cloud_edge_sla")
DEFAULT_POLICIES = (
    "random",
    "local_only",
    "best_uplink",
    "largest_queue",
    "nearest_edge",
    "least_loaded_edge",
    "cloud_edge",
)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_eval(
    *,
    python_exe: str,
    scenario: str,
    policy: str,
    seed: int,
    episodes: int,
    trace: Path,
    output: Path,
) -> subprocess.CompletedProcess:
    command = [
        python_exe,
        "eval_baselines.py",
        "--scenario",
        scenario,
        "--trace",
        str(trace),
        "--episodes",
        str(episodes),
        "--seed",
        str(seed),
        "--policy",
        policy,
        "--output",
        str(output),
    ]
    return subprocess.run(command, text=True, capture_output=True, check=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run formal multi-seed heuristic baselines for MEC scenarios.")
    parser.add_argument(
        "--trace",
        type=Path,
        default=Path("experiment_records/trace_profiles/alibaba_light_normal_urgent.csv"),
        help="Typed trace profile CSV.",
    )
    parser.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS), choices=SCENARIO_NAMES)
    parser.add_argument("--policies", nargs="+", default=list(DEFAULT_POLICIES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27, 37, 47])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiment_records/formal_baselines/heuristic_trace_profile"),
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable for eval_baselines.py.")
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
            for policy in args.policies:
                output = args.output_root / scenario / f"{policy}_seed{seed}.csv"
                output.parent.mkdir(parents=True, exist_ok=True)
                result = run_eval(
                    python_exe=args.python,
                    scenario=scenario,
                    policy=policy,
                    seed=seed,
                    episodes=args.episodes,
                    trace=args.trace,
                    output=output,
                )
                rows = read_csv(output) if output.exists() else []
                record = {
                    "scenario": scenario,
                    "policy": policy,
                    "seed": seed,
                    "episodes": args.episodes,
                    "trace": str(args.trace),
                    "output": str(output),
                    "returncode": result.returncode,
                    "stdout_tail": result.stdout.strip().splitlines()[-5:],
                    "stderr_tail": result.stderr.strip().splitlines()[-5:],
                    "rows": rows,
                }
                runs.append(record)
                status = "ok" if result.returncode == 0 else "failed"
                print(f"{status} scenario={scenario} policy={policy} seed={seed} output={output}")
                if result.returncode != 0:
                    failures.append(record)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "trace": str(args.trace),
        "scenarios": list(args.scenarios),
        "policies": list(args.policies),
        "seeds": list(args.seeds),
        "episodes": args.episodes,
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
