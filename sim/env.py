from __future__ import annotations

import math
import random
from dataclasses import asdict

from .config import MECConfig
from .entities import MECServer, Task, User


class MECEnv:
    """
    Minimal slotted MEC simulator.

    Action:
        A sequence of user ids to offload this step. The first
        `config.max_offloads_per_step` valid ids are accepted.
    Observation:
        A dictionary with per-user numeric features and global server state.
    """

    def __init__(self, config: MECConfig | None = None):
        self.config = config or MECConfig()
        self.rng = random.Random(self.config.random_seed)
        self.users: list[User] = []
        self.server = MECServer()
        self.current_step = 0
        self.next_task_id = 0
        self.last_info: dict = {}

    def reset(self, seed: int | None = None) -> tuple[dict, dict]:
        if seed is not None:
            self.rng.seed(seed)
        elif self.config.random_seed is not None:
            self.rng.seed(self.config.random_seed)

        self.current_step = 0
        self.next_task_id = 0
        self.server = MECServer()
        self.users = []
        for user_id in range(self.config.num_users):
            position = self.rng.uniform(0.0, self.config.area_size)
            velocity = self.rng.uniform(-4.0, 4.0)
            if abs(velocity) < 0.5:
                velocity = 1.5
            self.users.append(User(user_id=user_id, position=position, velocity=velocity))

        obs = self._build_observation()
        info = self._build_info(completed=0, dropped=0, avg_delay=0.0)
        self.last_info = info
        return obs, info

    def step(self, action: list[int] | tuple[int, ...] | None) -> tuple[dict, float, bool, dict]:
        self.current_step += 1
        accepted = self._normalize_action(action)

        self._move_users()
        self._generate_tasks()
        self._assign_offloads(accepted)
        completed_locally = self._process_local_compute()
        completed_on_mec = self._process_mec_pipeline()
        dropped = self._drop_expired_tasks()

        completed = completed_locally + completed_on_mec
        avg_delay = (
            sum(self.current_step - task.created_step for task in completed) / len(completed)
            if completed
            else 0.0
        )
        reward = self._compute_reward(avg_delay=avg_delay, completed=len(completed), dropped=dropped)
        done = self.current_step >= self.config.episode_length
        info = self._build_info(completed=len(completed), dropped=dropped, avg_delay=avg_delay)
        info["accepted_action"] = accepted
        self.last_info = info
        return self._build_observation(), reward, done, info

    def sample_random_action(self) -> list[int]:
        count = self.rng.randint(0, self.config.max_offloads_per_step)
        user_ids = list(range(self.config.num_users))
        self.rng.shuffle(user_ids)
        return user_ids[:count]

    def get_config(self) -> dict:
        return asdict(self.config)

    def _normalize_action(self, action: list[int] | tuple[int, ...] | None) -> list[int]:
        if action is None:
            return []
        unique: list[int] = []
        seen: set[int] = set()
        for raw_id in action:
            if not isinstance(raw_id, int):
                continue
            if raw_id < 0 or raw_id >= self.config.num_users:
                continue
            if raw_id in seen:
                continue
            seen.add(raw_id)
            unique.append(raw_id)
            if len(unique) >= self.config.max_offloads_per_step:
                break
        return unique

    def _move_users(self) -> None:
        for user in self.users:
            user.position += user.velocity * self.config.step_duration
            if user.position < 0.0:
                user.position = -user.position
                user.velocity *= -1.0
            elif user.position > self.config.area_size:
                user.position = 2 * self.config.area_size - user.position
                user.velocity *= -1.0

    def _generate_tasks(self) -> None:
        for user in self.users:
            if self.rng.random() > self.config.task_arrival_prob:
                continue
            size = self.rng.uniform(self.config.task_size_min, self.config.task_size_max)
            cycles = size * self.config.task_cycles_per_unit
            task = Task(
                task_id=self.next_task_id,
                size=size,
                total_cycles=cycles,
                remaining_cycles=cycles,
                created_step=self.current_step,
                deadline_step=self.current_step + self.config.task_deadline,
                remaining_upload=size,
            )
            self.next_task_id += 1
            user.queue.append(task)

    def _assign_offloads(self, accepted: list[int]) -> None:
        for user_id in accepted:
            user = self.users[user_id]
            if not user.queue:
                continue
            task = user.queue[0]
            if task.assigned_to_mec:
                continue
            task.assigned_to_mec = True

    def _process_local_compute(self) -> list[Task]:
        completed: list[Task] = []
        for user in self.users:
            if not user.queue:
                continue
            task = user.queue[0]
            if task.assigned_to_mec:
                continue
            task.remaining_cycles -= self.config.local_compute_rate * self.config.step_duration
            if task.remaining_cycles <= 0:
                completed.append(user.queue.pop(0))
        return completed

    def _process_mec_pipeline(self) -> list[Task]:
        newly_uploaded: list[Task] = []
        for user in self.users:
            if not user.queue:
                continue
            task = user.queue[0]
            if not task.assigned_to_mec or task.uploaded:
                continue
            upload_rate = self._uplink_rate(user.position, noisy=True)
            task.remaining_upload -= upload_rate * self.config.step_duration
            if task.remaining_upload <= 0:
                task.uploaded = True
                newly_uploaded.append(task)

        for task in newly_uploaded:
            self.server.queue.append(task)

        completed: list[Task] = []
        budget = self.config.mec_compute_rate * self.config.step_duration
        while self.server.queue and budget > 0:
            task = self.server.queue[0]
            consume = min(task.remaining_cycles, budget)
            task.remaining_cycles -= consume
            budget -= consume
            if task.remaining_cycles <= 0:
                self.server.queue.pop(0)
                completed.append(task)
                self._remove_completed_task_from_user(task.task_id)
        return completed

    def _remove_completed_task_from_user(self, task_id: int) -> None:
        for user in self.users:
            if user.queue and user.queue[0].task_id == task_id:
                user.queue.pop(0)
                return

    def _drop_expired_tasks(self) -> int:
        dropped = 0
        expired_server_ids = {
            task.task_id for task in self.server.queue if self.current_step >= task.deadline_step
        }
        if expired_server_ids:
            self.server.queue = [task for task in self.server.queue if task.task_id not in expired_server_ids]

        for user in self.users:
            keep: list[Task] = []
            for task in user.queue:
                if self.current_step >= task.deadline_step:
                    dropped += 1
                else:
                    keep.append(task)
            user.queue = keep
        return dropped

    def _uplink_rate(self, position: float, noisy: bool = False) -> float:
        rsu_position = self.config.area_size / 2.0
        distance = abs(position - rsu_position)
        base = self.config.base_uplink_rate / (1.0 + distance / self.config.pathloss_bias)
        if noisy:
            noise = 1.0 + self.rng.uniform(-self.config.channel_noise, self.config.channel_noise)
            base *= noise
        return max(0.5, base)

    def _compute_reward(self, avg_delay: float, completed: int, dropped: int) -> float:
        total_queue = sum(len(user.queue) for user in self.users) + len(self.server.queue)
        return (
            -self.config.delay_penalty * avg_delay
            -self.config.drop_penalty * dropped
            -self.config.queue_penalty * total_queue
            + 0.2 * completed
        )

    def _build_observation(self) -> dict:
        per_user = []
        for user in self.users:
            current_rate = self._uplink_rate(user.position, noisy=False)
            current_task_size = user.queue[0].size if user.queue else 0.0
            current_task_cycles = user.queue[0].remaining_cycles if user.queue else 0.0
            per_user.append(
                {
                    "user_id": user.user_id,
                    "position": round(user.position, 3),
                    "velocity": round(user.velocity, 3),
                    "queue_length": len(user.queue),
                    "current_task_size": round(current_task_size, 3),
                    "current_task_remaining_cycles": round(current_task_cycles, 3),
                    "uplink_rate": round(current_rate, 3),
                }
            )
        return {
            "step": self.current_step,
            "users": per_user,
            "server_queue_length": len(self.server.queue),
            "max_offloads_per_step": self.config.max_offloads_per_step,
        }

    def _build_info(self, completed: int, dropped: int, avg_delay: float) -> dict:
        total_queue = sum(len(user.queue) for user in self.users) + len(self.server.queue)
        return {
            "step": self.current_step,
            "completed_tasks": completed,
            "dropped_tasks": dropped,
            "avg_delay": round(avg_delay, 3),
            "total_queue": total_queue,
        }
