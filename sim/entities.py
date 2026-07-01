from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Task:
    task_id: int
    size: float
    total_cycles: float
    remaining_cycles: float
    created_step: int
    deadline_step: int
    task_type: str = "type_0"
    task_type_id: int = 0
    output_size: float = 0.0
    priority: float = 1.0
    assigned_to_mec: bool = False
    edge_server_id: int | None = None
    uploaded: bool = False
    remaining_upload: float = 0.0


@dataclass
class User:
    user_id: int
    position: float
    velocity: float
    queue: list[Task] = field(default_factory=list)


@dataclass
class MECServer:
    server_id: int = 0
    position: float = 0.0
    compute_rate: float = 14.0
    coverage_radius: float | None = None
    queue: list[Task] = field(default_factory=list)
