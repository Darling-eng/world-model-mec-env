from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path


RESOURCE_COLUMNS = ("cpu", "io", "bandwidth", "ram")


def iter_rows(sql_path: Path, table: str):
    marker = f"INSERT INTO `{table}` VALUES "
    with sql_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if marker not in line:
                continue
            values = line.split(marker, 1)[1].strip()
            yield from iter_tuple_literals(values)


def iter_tuple_literals(values: str):
    depth = 0
    start = None
    for index, char in enumerate(values):
        if char == "(":
            if depth == 0:
                start = index
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0 and start is not None:
                yield values[start : index + 1]
                start = None
        elif char == ";" and depth == 0:
            break


def normalize(value: float, min_value: float, max_value: float, low: float, high: float) -> float:
    if max_value <= min_value:
        return (low + high) / 2.0
    ratio = (value - min_value) / (max_value - min_value)
    return low + ratio * (high - low)


def convert_rows(rows: list[tuple], args: argparse.Namespace) -> list[dict]:
    timestamps = [float(row[5]) for row in rows]
    longitudes = [float(row[8]) for row in rows]
    demands = [sum(float(row[idx]) for idx in range(1, 5)) for row in rows]
    min_ts = min(timestamps)
    min_lon, max_lon = min(longitudes), max(longitudes)
    min_demand, max_demand = min(demands), max(demands)

    converted = []
    for index, row in enumerate(rows):
        demand = demands[index]
        step = 1 + int((float(row[5]) - min_ts) / args.time_scale)
        if step > args.episode_length:
            continue
        size = normalize(demand, min_demand, max_demand, args.min_size, args.max_size)
        position = normalize(float(row[8]), min_lon, max_lon, 0.0, args.area_size)
        cycles = max(1.0, float(row[6]) * args.duration_scale)
        converted.append(
            {
                "step": step,
                "user_id": index % args.num_users,
                "size": round(size, 4),
                "cycles": round(cycles, 4),
                "deadline": args.deadline,
                "upload": round(size, 4),
                "position": round(position, 4),
                "source_id": int(row[0]),
            }
        )
    return converted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a normalized MEC task trace from edge_task_scheduling.sql.")
    parser.add_argument("--sql", type=Path, required=True, help="Path to edge_task_scheduling.sql.")
    parser.add_argument("--table", default="alibaba_cluster_trace", help="SQL table to extract.")
    parser.add_argument("--output", type=Path, required=True, help="Output normalized CSV path.")
    parser.add_argument("--max-rows", type=int, default=500, help="Maximum rows to parse from the table.")
    parser.add_argument("--num-users", type=int, default=10, help="Number of simulated MEC users.")
    parser.add_argument("--episode-length", type=int, default=100, help="Maximum simulator step to keep.")
    parser.add_argument("--time-scale", type=float, default=2.0, help="Trace timestamp units per simulator step.")
    parser.add_argument("--duration-scale", type=float, default=1.0, help="Scale SQL duration to compute cycles.")
    parser.add_argument("--deadline", type=int, default=12, help="Relative deadline in simulator steps.")
    parser.add_argument("--min-size", type=float, default=4.0, help="Minimum normalized task upload size.")
    parser.add_argument("--max-size", type=float, default=10.0, help="Maximum normalized task upload size.")
    parser.add_argument("--area-size", type=float, default=100.0, help="Simulator area size for position scaling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for tuple_text in iter_rows(args.sql, args.table):
        rows.append(ast.literal_eval(tuple_text))
        if len(rows) >= args.max_rows:
            break
    if not rows:
        raise SystemExit(f"No rows found for table: {args.table}")

    converted = convert_rows(rows, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["step", "user_id", "size", "cycles", "deadline", "upload", "position", "source_id"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(converted)
    print(f"wrote_trace={args.output} source_rows={len(rows)} kept_rows={len(converted)} table={args.table}")


if __name__ == "__main__":
    main()
