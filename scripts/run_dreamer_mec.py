from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run DreamerV3 with the MEC environment bridge."
    )
    parser.add_argument(
        "--dreamer-dir",
        required=True,
        help="Absolute path to the cloned DreamerV3 repository.",
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=None,
        help="Optional CSV trace path for trace-driven MEC task arrivals.",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Optional named MEC scenario, for example multi_edge_network_sla or cloud_edge_sla.",
    )
    parser.add_argument(
        "--reward-preset",
        choices=["debug", "sla"],
        default="debug",
        help="Reward preset. debug preserves old smoke runs; sla emphasizes completion and deadline safety.",
    )
    args, remaining = parser.parse_known_args()
    return args, remaining


def main() -> None:
    args, remaining = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    dreamer_dir = Path(args.dreamer_dir).resolve()

    if not dreamer_dir.exists():
        raise FileNotFoundError(f"DreamerV3 directory not found: {dreamer_dir}")
    if args.trace is not None:
        trace_path = args.trace.resolve()
        if not trace_path.exists():
            raise FileNotFoundError(f"MEC trace file not found: {trace_path}")
        os.environ["MEC_TRACE_PATH"] = str(trace_path)
    if args.scenario is not None:
        os.environ["MEC_SCENARIO"] = args.scenario
    os.environ["MEC_REWARD_PRESET"] = args.reward_preset

    sys.path.insert(0, str(repo_root))
    from sim import register_dreamer_envs

    register_dreamer_envs()
    print(
        "MEC Dreamer config: "
        f"scenario={os.environ.get('MEC_SCENARIO', '<custom>')} "
        f"trace={os.environ.get('MEC_TRACE_PATH', '<synthetic>')} "
        f"reward_preset={os.environ['MEC_REWARD_PRESET']}"
    )

    sys.path.insert(0, str(dreamer_dir))
    from dreamerv3 import main as dreamer_main

    sys.argv = [str(dreamer_dir / "dreamerv3" / "main.py"), *remaining]
    dreamer_main.main()


if __name__ == "__main__":
    main()
