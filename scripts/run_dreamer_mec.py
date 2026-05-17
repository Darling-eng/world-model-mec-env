from __future__ import annotations

import argparse
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
    args, remaining = parser.parse_known_args()
    return args, remaining


def main() -> None:
    args, remaining = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    dreamer_dir = Path(args.dreamer_dir).resolve()

    if not dreamer_dir.exists():
        raise FileNotFoundError(f"DreamerV3 directory not found: {dreamer_dir}")

    sys.path.insert(0, str(repo_root))
    from sim import register_dreamer_envs

    register_dreamer_envs()

    sys.path.insert(0, str(dreamer_dir))
    from dreamerv3 import main as dreamer_main

    sys.argv = [str(dreamer_dir / "dreamerv3" / "main.py"), *remaining]
    dreamer_main.main()


if __name__ == "__main__":
    main()
