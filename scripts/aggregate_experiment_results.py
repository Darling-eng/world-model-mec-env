from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path


METRIC_KEYS = [
    "total_reward",
    "completed_tasks",
    "dropped_tasks",
    "deadline_violations",
    "deadline_violation_rate",
    "avg_delay",
    "avg_total_queue",
]


def expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    return sorted(dict.fromkeys(path.resolve() for path in paths))


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_number(value) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def infer_algorithm(path: Path, row: dict) -> str:
    if row.get("algorithm"):
        return str(row["algorithm"])
    if row.get("policy"):
        return str(row["policy"])
    text = str(path).lower()
    if "ppo" in text:
        return "ppo"
    if "dreamer" in text:
        return "dreamerv3"
    if "sac" in text:
        return "sac"
    if "dqn" in text:
        return "dqn"
    return path.stem


def normalize_row(path: Path, row: dict, *, phase: str = "eval") -> dict:
    algorithm = infer_algorithm(path, row)
    output = {
        "source_file": str(path),
        "algorithm": algorithm,
        "phase": str(row.get("phase") or phase),
        "episodes": str(row.get("episodes") or ""),
        "seed": str(row.get("seed") or ""),
    }
    for key in METRIC_KEYS:
        output[key] = normalize_number(row.get(key))
    return output


def select_jsonl_rows(rows: list[dict]) -> list[dict]:
    eval_rows = [row for row in rows if row.get("phase") == "eval"]
    if eval_rows:
        return eval_rows
    return rows[-1:] if rows else []


def load_result_rows(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return [normalize_row(path, row) for row in select_jsonl_rows(read_jsonl(path))]
    if suffix == ".csv":
        return [normalize_row(path, row) for row in read_csv(path)]
    raise ValueError(f"Unsupported result file: {path}")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["algorithm", "phase", "episodes", "seed", *METRIC_KEYS, "source_file"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate MEC baseline, PPO, SAC, and DreamerV3 result files.")
    parser.add_argument("--inputs", nargs="+", required=True, help="CSV/JSONL result files or glob patterns.")
    parser.add_argument("--output", type=Path, required=True, help="Output summary CSV path.")
    args = parser.parse_args()

    rows = []
    for path in expand_inputs(args.inputs):
        if not path.exists():
            raise FileNotFoundError(path)
        rows.extend(load_result_rows(path))
    write_csv(args.output, rows)
    print(f"wrote_summary={args.output} rows={len(rows)}")


if __name__ == "__main__":
    main()
