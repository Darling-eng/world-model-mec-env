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
    task_type: str | None = None
    task_type_id: int | None = None
    output_size: float | None = None
    priority: float | None = None
    position: float | None = None


def load_trace_tasks(
    path: str | Path,
    *,
    num_users: int,
    default_deadline: int,
    cycles_per_unit: float,
    task_type_count: int = 1,
) -> dict[int, list[TraceTaskSpec]]:
    """Load normalized task traces keyed by simulator step."""
    trace_path = Path(path)
    tasks_by_step: dict[int, list[TraceTaskSpec]] = {}
    with trace_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            task = _row_to_task(
                row,
                num_users=num_users,
                default_deadline=default_deadline,
                cycles_per_unit=cycles_per_unit,
                task_type_count=task_type_count,
            )
            tasks_by_step.setdefault(task.step, []).append(task)
    return tasks_by_step


def _row_to_task(
    row: dict[str, str],
    *,
    num_users: int,
    default_deadline: int,
    cycles_per_unit: float,
    task_type_count: int,
) -> TraceTaskSpec:
    step = max(1, int(float(row["step"])))
    user_id = int(float(row.get("user_id") or 0)) % num_users
    size = max(0.001, float(row["size"]))
    cycles = float(row.get("cycles") or row.get("total_cycles") or 0.0)
    if cycles <= 0.0:
        cycles = size * cycles_per_unit
    deadline = int(float(row.get("deadline") or default_deadline))
    upload = float(row.get("upload") or size)
    raw_task_type = row.get("task_type") or row.get("type")
    task_type_id, task_type = _parse_task_type(raw_task_type, task_type_count)
    raw_output_size = row.get("output_size") or row.get("result_size")
    output_size = float(raw_output_size) if raw_output_size not in (None, "") else None
    raw_priority = row.get("priority")
    priority = float(raw_priority) if raw_priority not in (None, "") else None
    raw_position = row.get("position")
    position = float(raw_position) if raw_position not in (None, "") else None
    return TraceTaskSpec(
        step=step,
        user_id=user_id,
        size=size,
        cycles=max(0.001, cycles),
        deadline=max(1, deadline),
        upload=max(0.001, upload),
        task_type=task_type,
        task_type_id=task_type_id,
        output_size=output_size if output_size is None else max(0.0, output_size),
        priority=priority if priority is None else max(0.0, priority),
        position=position,
    )


def _parse_task_type(raw_task_type: str | None, task_type_count: int) -> tuple[int, str]:
    count = max(1, int(task_type_count))
    if raw_task_type in (None, ""):
        return 0, "type_0"
    value = raw_task_type.strip()
    try:
        raw_id = int(float(value))
        return raw_id % count, f"type_{raw_id}"
    except ValueError:
        stable_id = sum(ord(char) for char in value) % count
        return stable_id, value
