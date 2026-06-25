from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TraceTaskSpec:
    step: int
    user_id: int
    size: float
    cycles: float
    deadline: int
    upload: float
    position: float | None = None


def load_trace_tasks(
    path: str | Path,
    *,
    num_users: int,
    default_deadline: int,
    cycles_per_unit: float,
) -> dict[int, list[TraceTaskSpec]]:
    """Load normalized task traces keyed by simulator step."""
    trace_path = Path(path)
    tasks_by_step: dict[int, list[TraceTaskSpec]] = {}
    with trace_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            task = _row_to_task(
                row,
                num_users=num_users,
                default_deadline=default_deadline,
                cycles_per_unit=cycles_per_unit,
            )
            tasks_by_step.setdefault(task.step, []).append(task)
    return tasks_by_step


def _row_to_task(
    row: dict[str, str],
    *,
    num_users: int,
    default_deadline: int,
    cycles_per_unit: float,
) -> TraceTaskSpec:
    step = max(1, int(float(row["step"])))
    user_id = int(float(row.get("user_id") or 0)) % num_users
    size = max(0.001, float(row["size"]))
    cycles = float(row.get("cycles") or row.get("total_cycles") or 0.0)
    if cycles <= 0.0:
        cycles = size * cycles_per_unit
    deadline = int(float(row.get("deadline") or default_deadline))
    upload = float(row.get("upload") or size)
    raw_position = row.get("position")
    position = float(raw_position) if raw_position not in (None, "") else None
    return TraceTaskSpec(
        step=step,
        user_id=user_id,
        size=size,
        cycles=max(0.001, cycles),
        deadline=max(1, deadline),
        upload=max(0.001, upload),
        position=position,
    )
