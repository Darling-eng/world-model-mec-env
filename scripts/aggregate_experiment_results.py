from __future__ import annotations

import argparse
import csv
import glob
import json
import math
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


def to_float(value: str) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def summarize_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (str(row.get("algorithm", "")), str(row.get("phase", "")))
        groups.setdefault(key, []).append(row)

    summary_rows = []
    for (algorithm, phase), group_rows in sorted(groups.items()):
        seeds = sorted({str(row.get("seed", "")) for row in group_rows if str(row.get("seed", ""))})
        output = {
            "algorithm": algorithm,
            "phase": phase,
            "runs": str(len(group_rows)),
            "seeds": ";".join(seeds),
        }
        for key in METRIC_KEYS:
            values = [value for row in group_rows if (value := to_float(row.get(key, ""))) is not None]
            output[f"{key}_mean"] = f"{mean(values):.6f}" if values else ""
            output[f"{key}_std"] = f"{sample_std(values):.6f}" if values else ""
        summary_rows.append(output)
    return summary_rows


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_fields = [field for key in METRIC_KEYS for field in (f"{key}_mean", f"{key}_std")]
    fieldnames = ["algorithm", "phase", "runs", "seeds", *metric_fields]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate MEC baseline, PPO, SAC, and DreamerV3 result files.")
    parser.add_argument("--inputs", nargs="+", required=True, help="CSV/JSONL result files or glob patterns.")
    parser.add_argument("--output", type=Path, required=True, help="Output summary CSV path.")
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional mean/std CSV grouped by algorithm and phase.",
    )
    args = parser.parse_args()

    rows = []
    for path in expand_inputs(args.inputs):
        if not path.exists():
            raise FileNotFoundError(path)
        rows.extend(load_result_rows(path))
    write_csv(args.output, rows)
    if args.summary_output is not None:
        write_summary_csv(args.summary_output, summarize_rows(rows))
    print(f"wrote_summary={args.output} rows={len(rows)}")
    if args.summary_output is not None:
        print(f"wrote_summary_stats={args.summary_output}")


if __name__ == "__main__":
    main()
