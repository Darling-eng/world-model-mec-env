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
    assigned_to_mec: bool = False
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
    queue: list[Task] = field(default_factory=list)
