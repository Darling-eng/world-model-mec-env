from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


PROFILE_NAMES = ("light_normal_urgent",)


def classify_size(size: float, low: float, high: float) -> tuple[int, str]:
    if size <= low:
        return 0, "light"
    if size <= high:
        return 1, "normal"
    return 2, "urgent"


def output_ratio_for_type(task_type_id: int) -> float:
    return (0.05, 0.1, 0.2)[task_type_id]


def priority_for_type(task_type_id: int) -> float:
    return (1.0, 2.0, 5.0)[task_type_id]


def deadline_for_type(task_type_id: int) -> int:
    return (14, 10, 6)[task_type_id]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def build_profile_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict]:
    sizes = [float(row["size"]) for row in rows if row.get("size")]
    low = percentile(sizes, 1 / 3)
    high = percentile(sizes, 2 / 3)
    profiled = []
    counts: dict[str, int] = {"light": 0, "normal": 0, "urgent": 0}
    for row in rows:
        output = dict(row)
        size = max(0.001, float(row["size"]))
        task_type_id, task_type = classify_size(size, low, high)
        output["task_type_id"] = str(task_type_id)
        output["task_type"] = task_type
        output["deadline"] = str(deadline_for_type(task_type_id))
        output["output_size"] = f"{size * output_ratio_for_type(task_type_id):.6f}"
        output["priority"] = f"{priority_for_type(task_type_id):.1f}"
        profiled.append(output)
        counts[task_type] += 1
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": "light_normal_urgent",
        "size_thresholds": {"light_max": low, "normal_max": high},
        "task_type_count": 3,
        "task_type_names": ["light", "normal", "urgent"],
        "task_type_counts": counts,
        "task_deadlines_by_type": [14, 10, 6],
        "task_output_ratios_by_type": [0.05, 0.1, 0.2],
        "task_priorities_by_type": [1.0, 2.0, 5.0],
    }
    return profiled, manifest


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step",
        "user_id",
        "size",
        "cycles",
        "deadline",
        "upload",
        "position",
        "source_id",
        "task_type_id",
        "task_type",
        "output_size",
        "priority",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a typed MEC trace profile from a normalized trace CSV.")
    parser.add_argument("--input", type=Path, required=True, help="Input normalized trace CSV.")
    parser.add_argument("--output", type=Path, required=True, help="Output typed trace CSV.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional JSON manifest path. Defaults to <output>.manifest.json.",
    )
    parser.add_argument("--profile", choices=PROFILE_NAMES, default="light_normal_urgent")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    profiled, manifest = build_profile_rows(rows)
    manifest["input"] = str(args.input)
    manifest["output"] = str(args.output)
    manifest["row_count"] = len(profiled)
    write_rows(args.output, profiled)
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote_trace_profile={args.output} rows={len(profiled)}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
