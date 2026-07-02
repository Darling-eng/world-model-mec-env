from __future__ import annotations

import math
import random
from dataclasses import asdict

from .config import MECConfig
from .entities import CloudServer, MECServer, Task, User
from .trace import TraceTaskSpec, load_trace_tasks


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
        self.servers: list[MECServer] = self._build_servers()
        self.server = self.servers[0]
        self.cloud = self._build_cloud()
        self.current_step = 0
        self.next_task_id = 0
        self.last_info: dict = {}
        self.last_step_metrics: dict = {}
        self.trace_tasks = self._load_trace_tasks()

    def reset(self, seed: int | None = None) -> tuple[dict, dict]:
        if seed is not None:
            self.rng.seed(seed)
        elif self.config.random_seed is not None:
            self.rng.seed(self.config.random_seed)

        self.current_step = 0
        self.next_task_id = 0
        self.last_step_metrics = {}
        self.servers = self._build_servers()
        self.server = self.servers[0]
        self.cloud = self._build_cloud()
        self.users = []
        for user_id in range(self.config.num_users):
            position = self.rng.uniform(0.0, self.config.area_size)
            velocity = self.rng.uniform(-4.0, 4.0)
            if abs(velocity) < 0.5:
                velocity = 1.5
            self.users.append(User(user_id=user_id, position=position, velocity=velocity))

        obs = self._build_observation()
        info = self._build_info(completed=0, deadline_violations=0, avg_delay=0.0)
        self.last_info = info
        return obs, info

    def step(self, action: list | tuple | None) -> tuple[dict, float, bool, dict]:
        self.current_step += 1
        self._reset_step_metrics()
        accepted = self._normalize_action(action)
        accepted_users = [user_id for user_id, _ in accepted]

        self._move_users()
        self._generate_tasks()
        self._assign_offloads(accepted)
        completed_locally = self._process_local_compute()
        completed_on_mec = self._process_mec_pipeline()
        completed = completed_locally + completed_on_mec
        dropped_tasks = self._drop_expired_tasks()
        deadline_violations = len(dropped_tasks)
        avg_delay = (
            sum(self.current_step - task.created_step for task in completed) / len(completed)
            if completed
            else 0.0
        )
        reward = self._compute_reward(
            avg_delay=avg_delay,
            completed=len(completed),
            dropped=deadline_violations,
        )
        done = self.current_step >= self.config.episode_length
        info = self._build_info(
            completed=len(completed),
            deadline_violations=deadline_violations,
            avg_delay=avg_delay,
            completed_by_type=self._count_by_task_type(completed),
            dropped_by_type=self._count_by_task_type(dropped_tasks),
        )
        info["accepted_action"] = accepted_users
        info["accepted_offloads"] = [
            {"user_id": user_id, "edge_server_id": edge_server_id}
            for user_id, edge_server_id in accepted
        ]
        self.last_info = info
        return self._build_observation(), reward, done, info

    def sample_random_action(self) -> list[int]:
        count = self.rng.randint(0, self.config.max_offloads_per_step)
        user_ids = list(range(self.config.num_users))
        self.rng.shuffle(user_ids)
        return user_ids[:count]

    def get_config(self) -> dict:
        return asdict(self.config)

    def _reset_step_metrics(self) -> None:
        self.last_step_metrics = {
            "uplink_data": 0.0,
            "downlink_data": 0.0,
            "local_compute_used": 0.0,
            "cloud_compute_used": 0.0,
            "cloud_compute_capacity": (
                self.cloud.compute_rate * self.config.step_duration
                if self.config.enable_cloud_fallback
                else 0.0
            ),
            "cloud_completed_tasks": 0,
            "edge_compute_used": [0.0 for _ in self.servers],
            "edge_compute_capacity": [
                server.compute_rate * self.config.step_duration
                for server in self.servers
            ],
        }

    def _build_servers(self) -> list[MECServer]:
        server_count = max(1, int(self.config.num_edge_servers))
        positions = self._edge_positions(server_count)
        compute_rates = self._edge_compute_rates(server_count)
        return [
            MECServer(
                server_id=index,
                position=positions[index],
                compute_rate=compute_rates[index],
                coverage_radius=self.config.edge_server_coverage_radius,
            )
            for index in range(server_count)
        ]

    def _build_cloud(self) -> CloudServer:
        return CloudServer(compute_rate=self.config.cloud_compute_rate)

    def _edge_positions(self, server_count: int) -> list[float]:
        if self.config.edge_server_positions:
            configured = list(self.config.edge_server_positions)
            if len(configured) >= server_count:
                return configured[:server_count]
        if server_count == 1:
            return [self.config.area_size / 2.0]
        spacing = self.config.area_size / max(server_count - 1, 1)
        return [index * spacing for index in range(server_count)]

    def _edge_compute_rates(self, server_count: int) -> list[float]:
        if self.config.edge_server_compute_rates:
            configured = list(self.config.edge_server_compute_rates)
            if len(configured) >= server_count:
                return configured[:server_count]
        return [self.config.mec_compute_rate for _ in range(server_count)]

    def _normalize_action(self, action: list | tuple | None) -> list[tuple[int, int | None]]:
        if action is None:
            return []
        unique: list[tuple[int, int | None]] = []
        seen: set[int] = set()
        for raw_action in action:
            user_id, edge_server_id = self._parse_action_entry(raw_action)
            if user_id is None:
                continue
            if user_id < 0 or user_id >= self.config.num_users:
                continue
            if edge_server_id == -1 and not self.config.enable_cloud_fallback:
                continue
            if edge_server_id is not None and edge_server_id != -1 and not (0 <= edge_server_id < len(self.servers)):
                continue
            if user_id in seen:
                continue
            seen.add(user_id)
            unique.append((user_id, edge_server_id))
            if len(unique) >= self.config.max_offloads_per_step:
                break
        return unique

    def _parse_action_entry(self, raw_action) -> tuple[int | None, int | None]:
        if isinstance(raw_action, int):
            return raw_action, None
        if isinstance(raw_action, (list, tuple)) and len(raw_action) >= 1:
            user_raw = raw_action[0]
            server_raw = raw_action[1] if len(raw_action) >= 2 else None
            if not isinstance(user_raw, int):
                return None, None
            if server_raw is None:
                return user_raw, None
            if not isinstance(server_raw, int):
                return None, None
            return user_raw, server_raw
        return None, None

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
        if self.trace_tasks is not None:
            self._generate_trace_tasks()
            return

        for user in self.users:
            if self.rng.random() > self.config.task_arrival_prob:
                continue
            size = self.rng.uniform(self.config.task_size_min, self.config.task_size_max)
            task_type_id = self._sample_task_type_id()
            cycles = size * self._task_cycles_per_unit(task_type_id)
            task = self._make_task(
                user_id=user.user_id,
                size=size,
                cycles=cycles,
                deadline=self.current_step + self._task_deadline(task_type_id),
                upload=size,
                task_type_id=task_type_id,
            )
            user.queue.append(task)

    def _generate_trace_tasks(self) -> None:
        assert self.trace_tasks is not None
        for trace_task in self.trace_tasks.get(self.current_step, []):
            user = self.users[trace_task.user_id]
            if trace_task.position is not None:
                user.position = max(0.0, min(self.config.area_size, trace_task.position))
            task = self._make_task(
                user_id=user.user_id,
                size=trace_task.size,
                cycles=trace_task.cycles,
                deadline=self.current_step + trace_task.deadline,
                upload=trace_task.upload,
                task_type=trace_task.task_type,
                task_type_id=trace_task.task_type_id,
                output_size=trace_task.output_size,
                priority=trace_task.priority,
            )
            user.queue.append(task)

    def _make_task(
        self,
        *,
        user_id: int | None,
        size: float,
        cycles: float,
        deadline: int,
        upload: float,
        task_type: str | None = None,
        task_type_id: int | None = None,
        output_size: float | None = None,
        priority: float | None = None,
    ) -> Task:
        type_id = self._normalize_task_type_id(task_type_id)
        task = Task(
            task_id=self.next_task_id,
            user_id=user_id,
            size=size,
            total_cycles=cycles,
            remaining_cycles=cycles,
            created_step=self.current_step,
            deadline_step=deadline,
            task_type=task_type or self._task_type_name(type_id),
            task_type_id=type_id,
            output_size=(
                output_size
                if output_size is not None
                else size * self._task_output_ratio(type_id)
            ),
            priority=priority if priority is not None else self._task_priority(type_id),
            remaining_upload=upload,
            remaining_download=(
                output_size
                if output_size is not None
                else size * self._task_output_ratio(type_id)
            ),
        )
        self.next_task_id += 1
        return task

    def _sample_task_type_id(self) -> int:
        count = self._task_type_count()
        probabilities = self.config.task_type_probabilities
        if not probabilities:
            return self.rng.randrange(count)
        weights = [max(0.0, float(item)) for item in probabilities[:count]]
        if len(weights) < count:
            weights.extend([0.0] * (count - len(weights)))
        total = sum(weights)
        if total <= 0.0:
            return self.rng.randrange(count)
        pick = self.rng.random() * total
        cumulative = 0.0
        for index, weight in enumerate(weights):
            cumulative += weight
            if pick <= cumulative:
                return index
        return count - 1

    def _task_type_count(self) -> int:
        return max(1, int(self.config.task_type_count))

    def _normalize_task_type_id(self, task_type_id: int | None) -> int:
        if task_type_id is None:
            return 0
        return max(0, int(task_type_id)) % self._task_type_count()

    def _task_type_name(self, task_type_id: int) -> str:
        names = self.config.task_type_names
        if names and 0 <= task_type_id < len(names):
            return names[task_type_id]
        return f"type_{task_type_id}"

    def _task_cycles_per_unit(self, task_type_id: int) -> float:
        return float(
            self._task_profile_value(
                self.config.task_cycles_per_unit_by_type,
                task_type_id,
                self.config.task_cycles_per_unit,
            )
        )

    def _task_deadline(self, task_type_id: int) -> int:
        return max(
            1,
            int(
                self._task_profile_value(
                    self.config.task_deadlines_by_type,
                    task_type_id,
                    self.config.task_deadline,
                )
            ),
        )

    def _task_output_ratio(self, task_type_id: int) -> float:
        return max(
            0.0,
            float(
                self._task_profile_value(
                    self.config.task_output_ratios_by_type,
                    task_type_id,
                    0.1,
                )
            ),
        )

    def _task_priority(self, task_type_id: int) -> float:
        return max(
            0.0,
            float(
                self._task_profile_value(
                    self.config.task_priorities_by_type,
                    task_type_id,
                    1.0,
                )
            ),
        )

    def _task_profile_value(self, values, task_type_id: int, default):
        if values and 0 <= task_type_id < len(values):
            return values[task_type_id]
        return default

    def _load_trace_tasks(self) -> dict[int, list[TraceTaskSpec]] | None:
        if not self.config.task_trace_path:
            return None
        return load_trace_tasks(
            self.config.task_trace_path,
            num_users=self.config.num_users,
            default_deadline=self.config.task_deadline,
            cycles_per_unit=self.config.task_cycles_per_unit,
            task_type_count=self.config.task_type_count,
        )

    def _assign_offloads(self, accepted: list[tuple[int, int | None]]) -> None:
        for user_id, requested_server_id in accepted:
            user = self.users[user_id]
            if not user.queue:
                continue
            task = user.queue[0]
            if task.assigned_to_mec:
                continue
            task.assigned_to_mec = True
            if requested_server_id == -1:
                task.assigned_to_cloud = True
                task.edge_server_id = None
            else:
                task.edge_server_id = requested_server_id if requested_server_id is not None else self._select_edge_server(user)

    def _process_local_compute(self) -> list[Task]:
        completed: list[Task] = []
        for user in self.users:
            if not user.queue:
                continue
            task = user.queue[0]
            if task.assigned_to_mec:
                continue
            before_cycles = task.remaining_cycles
            task.remaining_cycles -= self.config.local_compute_rate * self.config.step_duration
            consumed = max(0.0, before_cycles - max(task.remaining_cycles, 0.0))
            self.last_step_metrics["local_compute_used"] += consumed
            if task.remaining_cycles <= 0:
                completed.append(user.queue.pop(0))
        return completed

    def _process_mec_pipeline(self) -> list[Task]:
        newly_uploaded: list[Task] = []
        upload_groups = self._active_upload_groups()
        for server_id, uploads in upload_groups.items():
            share = len(uploads) if self.config.enable_uplink_contention else 1
            for user, task in uploads:
                upload_rate = self._upload_rate_for_task(user.position, task, noisy=True) / share
                before_upload = task.remaining_upload
                task.remaining_upload -= upload_rate * self.config.step_duration
                self.last_step_metrics["uplink_data"] += max(0.0, before_upload - max(task.remaining_upload, 0.0))
                if task.remaining_upload <= 0:
                    task.uploaded = True
                    newly_uploaded.append(task)

        for task in newly_uploaded:
            if task.assigned_to_cloud:
                task.cloud_delay_remaining = max(0, int(self.config.cloud_wan_delay_steps))
                self.cloud.queue.append(task)
            else:
                self._server_for_task(task).queue.append(task)

        completed: list[Task] = []
        for server in self.servers:
            budget = server.compute_rate * self.config.step_duration
            while server.queue and budget > 0:
                task = server.queue[0]
                consume = min(task.remaining_cycles, budget)
                task.remaining_cycles -= consume
                budget -= consume
                self.last_step_metrics["edge_compute_used"][server.server_id] += consume
                if task.remaining_cycles <= 0:
                    server.queue.pop(0)
                    if self.config.enable_downlink_transmission:
                        task.remaining_download = max(task.remaining_download, 0.0)
                        task.downloaded = task.remaining_download <= 0.0
                        if task.downloaded:
                            completed.append(task)
                            self._remove_completed_task_from_user(task.task_id)
                        else:
                            server.downlink_queue.append(task)
                    else:
                        completed.append(task)
                        self._remove_completed_task_from_user(task.task_id)
        completed.extend(self._process_cloud_pipeline())
        completed.extend(self._process_downlink_pipeline())
        return completed

    def _active_upload_groups(self) -> dict[int, list[tuple[User, Task]]]:
        groups: dict[int, list[tuple[User, Task]]] = {}
        for user in self.users:
            if not user.queue:
                continue
            task = user.queue[0]
            if not task.assigned_to_mec or task.uploaded:
                continue
            server_id = -1 if task.assigned_to_cloud else self._server_for_task(task).server_id
            groups.setdefault(server_id, []).append((user, task))
        return groups

    def _process_downlink_pipeline(self) -> list[Task]:
        if not self.config.enable_downlink_transmission:
            return []
        completed: list[Task] = []
        for server in [*self.servers, self.cloud]:
            if not server.downlink_queue:
                continue
            active = [
                task for task in server.downlink_queue
                if task.user_id is not None and self._user_by_id(task.user_id) is not None
            ]
            share = max(len(active), 1)
            for task in active:
                user = self._user_by_id(task.user_id)
                if user is None:
                    continue
                downlink_rate = self._downlink_rate_for_task(user.position, task, noisy=True) / share
                before_download = task.remaining_download
                task.remaining_download -= downlink_rate * self.config.step_duration
                self.last_step_metrics["downlink_data"] += max(0.0, before_download - max(task.remaining_download, 0.0))
                if task.remaining_download <= 0:
                    task.downloaded = True
                    completed.append(task)
                    self._remove_completed_task_from_user(task.task_id)
            if completed:
                completed_ids = {task.task_id for task in completed}
                server.downlink_queue = [
                    task for task in server.downlink_queue
                    if task.task_id not in completed_ids
                ]
        return completed

    def _process_cloud_pipeline(self) -> list[Task]:
        if not self.config.enable_cloud_fallback:
            return []
        completed: list[Task] = []
        for task in self.cloud.queue:
            if task.cloud_delay_remaining > 0:
                task.cloud_delay_remaining -= 1
        budget = self.cloud.compute_rate * self.config.step_duration
        while self.cloud.queue and budget > 0:
            task = self.cloud.queue[0]
            if task.cloud_delay_remaining > 0:
                break
            consume = min(task.remaining_cycles, budget)
            task.remaining_cycles -= consume
            budget -= consume
            self.last_step_metrics["cloud_compute_used"] += consume
            if task.remaining_cycles <= 0:
                self.cloud.queue.pop(0)
                self.last_step_metrics["cloud_completed_tasks"] += 1
                if self.config.enable_downlink_transmission:
                    task.remaining_download = max(task.remaining_download, 0.0)
                    task.downloaded = task.remaining_download <= 0.0
                    if task.downloaded:
                        completed.append(task)
                        self._remove_completed_task_from_user(task.task_id)
                    else:
                        self.cloud.downlink_queue.append(task)
                else:
                    completed.append(task)
                    self._remove_completed_task_from_user(task.task_id)
        return completed

    def _user_by_id(self, user_id: int | None) -> User | None:
        if user_id is None:
            return None
        if 0 <= user_id < len(self.users):
            return self.users[user_id]
        return None

    def _server_for_task(self, task: Task) -> MECServer:
        if task.edge_server_id is None:
            return self.server
        if 0 <= task.edge_server_id < len(self.servers):
            return self.servers[task.edge_server_id]
        return self.server

    def _select_edge_server(self, user: User) -> int:
        candidates = self._reachable_servers(user.position)
        if not candidates:
            candidates = self.servers
        if self.config.edge_selection_policy == "least_loaded":
            best = min(
                candidates,
                key=lambda server: (
                    len(server.queue),
                    abs(user.position - server.position),
                    server.server_id,
                ),
            )
            return best.server_id
        if self.config.edge_selection_policy == "nearest":
            best = min(candidates, key=lambda server: (abs(user.position - server.position), server.server_id))
            return best.server_id
        raise ValueError(f"Unknown edge_selection_policy: {self.config.edge_selection_policy}")

    def _reachable_servers(self, position: float) -> list[MECServer]:
        reachable = []
        for server in self.servers:
            if server.coverage_radius is None or abs(position - server.position) <= server.coverage_radius:
                reachable.append(server)
        return reachable

    def _remove_completed_task_from_user(self, task_id: int) -> None:
        for user in self.users:
            if user.queue and user.queue[0].task_id == task_id:
                user.queue.pop(0)
                return

    def _drop_expired_tasks(self) -> list[Task]:
        dropped: list[Task] = []
        expired_server_ids = set()
        for server in self.servers:
            expired_server_tasks = [
                task for task in server.queue if self.current_step >= task.deadline_step
            ]
            expired_downlink_tasks = [
                task for task in server.downlink_queue if self.current_step >= task.deadline_step
            ]
            server_expired_ids = {
                task.task_id for task in [*expired_server_tasks, *expired_downlink_tasks]
            }
            dropped.extend(expired_server_tasks)
            dropped.extend(expired_downlink_tasks)
            expired_server_ids.update(server_expired_ids)
            if server_expired_ids:
                server.queue = [task for task in server.queue if task.task_id not in server_expired_ids]
                server.downlink_queue = [
                    task for task in server.downlink_queue
                    if task.task_id not in server_expired_ids
                ]
        cloud_expired = [
            task for task in [*self.cloud.queue, *self.cloud.downlink_queue]
            if self.current_step >= task.deadline_step
        ]
        cloud_expired_ids = {task.task_id for task in cloud_expired}
        if cloud_expired_ids:
            dropped.extend(cloud_expired)
            expired_server_ids.update(cloud_expired_ids)
            self.cloud.queue = [
                task for task in self.cloud.queue
                if task.task_id not in cloud_expired_ids
            ]
            self.cloud.downlink_queue = [
                task for task in self.cloud.downlink_queue
                if task.task_id not in cloud_expired_ids
            ]

        for user in self.users:
            keep: list[Task] = []
            for task in user.queue:
                if task.task_id in expired_server_ids:
                    continue
                if self.current_step >= task.deadline_step:
                    dropped.append(task)
                else:
                    keep.append(task)
            user.queue = keep
        return dropped

    def _uplink_rate(self, position: float, server_id: int = 0, noisy: bool = False) -> float:
        server = self.servers[min(max(server_id, 0), len(self.servers) - 1)]
        distance = abs(position - server.position)
        base = self.config.base_uplink_rate / (1.0 + distance / self.config.pathloss_bias)
        if noisy:
            noise = 1.0 + self.rng.uniform(-self.config.channel_noise, self.config.channel_noise)
            base *= noise
        return max(0.5, base)

    def _downlink_rate(self, position: float, server_id: int = 0, noisy: bool = False) -> float:
        server = self.servers[min(max(server_id, 0), len(self.servers) - 1)]
        distance = abs(position - server.position)
        base = self.config.base_downlink_rate / (1.0 + distance / self.config.pathloss_bias)
        if noisy:
            noise = 1.0 + self.rng.uniform(-self.config.channel_noise, self.config.channel_noise)
            base *= noise
        return max(0.5, base)

    def _upload_rate_for_task(self, position: float, task: Task, noisy: bool = False) -> float:
        if task.assigned_to_cloud:
            return self._cloud_uplink_rate(noisy=noisy)
        return self._uplink_rate(position, task.edge_server_id or 0, noisy=noisy)

    def _downlink_rate_for_task(self, position: float, task: Task, noisy: bool = False) -> float:
        if task.assigned_to_cloud:
            return self._cloud_downlink_rate(noisy=noisy)
        return self._downlink_rate(position, task.edge_server_id or 0, noisy=noisy)

    def _cloud_uplink_rate(self, noisy: bool = False) -> float:
        base = self.config.cloud_wan_upload_rate
        if noisy:
            base *= 1.0 + self.rng.uniform(-self.config.channel_noise, self.config.channel_noise)
        return max(0.25, base)

    def _cloud_downlink_rate(self, noisy: bool = False) -> float:
        base = self.config.cloud_wan_downlink_rate
        if noisy:
            base *= 1.0 + self.rng.uniform(-self.config.channel_noise, self.config.channel_noise)
        return max(0.25, base)

    def _compute_reward(self, avg_delay: float, completed: int, dropped: int) -> float:
        total_queue = self._total_queue_length()
        if self.config.reward_preset == "debug":
            return (
                -self.config.delay_penalty * avg_delay
                -self.config.drop_penalty * dropped
                -self.config.queue_penalty * total_queue
                + 0.2 * completed
            )
        if self.config.reward_preset == "sla":
            return (
                self.config.completion_bonus * completed
                - self.config.delay_penalty * avg_delay
                - self.config.sla_violation_penalty * dropped
                - self.config.queue_penalty * total_queue
            )
        raise ValueError(f"Unknown reward_preset: {self.config.reward_preset}")

    def _build_observation(self) -> dict:
        per_user = []
        for user in self.users:
            best_server = self._select_edge_server(user)
            current_rate = self._uplink_rate(user.position, best_server, noisy=False)
            current_task = user.queue[0] if user.queue else None
            current_task_size = current_task.size if current_task else 0.0
            current_task_cycles = current_task.remaining_cycles if current_task else 0.0
            current_task_type = current_task.task_type if current_task else ""
            current_task_type_id = current_task.task_type_id if current_task else 0
            current_task_output_size = current_task.output_size if current_task else 0.0
            current_task_priority = current_task.priority if current_task else 0.0
            current_task_deadline_remaining = (
                max(0, current_task.deadline_step - self.current_step)
                if current_task
                else 0
            )
            server_rates = [
                {
                    "server_id": server.server_id,
                    "uplink_rate": round(self._uplink_rate(user.position, server.server_id, noisy=False), 3),
                    "downlink_rate": round(self._downlink_rate(user.position, server.server_id, noisy=False), 3),
                    "reachable": server in self._reachable_servers(user.position),
                }
                for server in self.servers
            ]
            per_user.append(
                {
                    "user_id": user.user_id,
                    "position": round(user.position, 3),
                    "velocity": round(user.velocity, 3),
                    "queue_length": len(user.queue),
                    "current_task_size": round(current_task_size, 3),
                    "current_task_remaining_cycles": round(current_task_cycles, 3),
                    "current_task_type": current_task_type,
                    "current_task_type_id": current_task_type_id,
                    "current_task_output_size": round(current_task_output_size, 3),
                    "current_task_priority": round(current_task_priority, 3),
                    "current_task_deadline_remaining": current_task_deadline_remaining,
                    "uplink_rate": round(current_rate, 3),
                    "best_edge_server_id": best_server,
                    "server_rates": server_rates,
                }
            )
        servers = [
            {
                "server_id": server.server_id,
                "position": round(server.position, 3),
                "compute_rate": round(server.compute_rate, 3),
                "queue_length": len(server.queue),
                "downlink_queue_length": len(server.downlink_queue),
                "coverage_radius": server.coverage_radius,
            }
            for server in self.servers
        ]
        return {
            "step": self.current_step,
            "users": per_user,
            "servers": servers,
            "cloud": {
                "enabled": self.config.enable_cloud_fallback,
                "compute_rate": round(self.cloud.compute_rate, 3),
                "queue_length": len(self.cloud.queue),
                "downlink_queue_length": len(self.cloud.downlink_queue),
                "wan_upload_rate": round(self.config.cloud_wan_upload_rate, 3),
                "wan_downlink_rate": round(self.config.cloud_wan_downlink_rate, 3),
                "wan_delay_steps": self.config.cloud_wan_delay_steps,
            },
            "server_queue_length": sum(len(server.queue) for server in self.servers),
            "server_downlink_queue_length": sum(len(server.downlink_queue) for server in self.servers),
            "cloud_queue_length": len(self.cloud.queue),
            "cloud_downlink_queue_length": len(self.cloud.downlink_queue),
            "num_edge_servers": len(self.servers),
            "max_offloads_per_step": self.config.max_offloads_per_step,
            "enable_uplink_contention": self.config.enable_uplink_contention,
            "enable_downlink_transmission": self.config.enable_downlink_transmission,
            "enable_cloud_fallback": self.config.enable_cloud_fallback,
        }

    def _build_info(
        self,
        completed: int,
        deadline_violations: int,
        avg_delay: float,
        completed_by_type: dict[str, int] | None = None,
        dropped_by_type: dict[str, int] | None = None,
    ) -> dict:
        total_queue = self._total_queue_length()
        outcomes = completed + deadline_violations
        deadline_violation_rate = deadline_violations / outcomes if outcomes else 0.0
        active_uploads_by_server = {
            server_id: len(uploads)
            for server_id, uploads in self._active_upload_groups().items()
        }
        edge_compute_used = self.last_step_metrics.get("edge_compute_used", [0.0 for _ in self.servers])
        edge_compute_capacity = self.last_step_metrics.get("edge_compute_capacity", [0.0 for _ in self.servers])
        local_compute_used = float(self.last_step_metrics.get("local_compute_used", 0.0))
        cloud_compute_used = float(self.last_step_metrics.get("cloud_compute_used", 0.0))
        cloud_compute_capacity = float(self.last_step_metrics.get("cloud_compute_capacity", 0.0))
        uplink_data = float(self.last_step_metrics.get("uplink_data", 0.0))
        downlink_data = float(self.last_step_metrics.get("downlink_data", 0.0))
        edge_compute_total = sum(float(value) for value in edge_compute_used)
        energy_used = (
            local_compute_used * self.config.local_energy_per_cycle
            + edge_compute_total * self.config.edge_energy_per_cycle
            + cloud_compute_used * self.config.cloud_energy_per_cycle
            + (uplink_data + downlink_data) * self.config.network_energy_per_data
        )
        cloud_cost = cloud_compute_used * self.config.cloud_cost_per_cycle
        cloud_utilization = (
            cloud_compute_used / cloud_compute_capacity
            if cloud_compute_capacity > 0.0
            else 0.0
        )
        cloud_completed_tasks = int(self.last_step_metrics.get("cloud_completed_tasks", 0))
        edge_utilization = [
            used / capacity if capacity > 0 else 0.0
            for used, capacity in zip(edge_compute_used, edge_compute_capacity)
        ]
        return {
            "step": self.current_step,
            "completed_tasks": completed,
            "dropped_tasks": deadline_violations,
            "deadline_violations": deadline_violations,
            "deadline_violation_rate": round(deadline_violation_rate, 3),
            "avg_delay": round(avg_delay, 3),
            "total_queue": total_queue,
            "server_queue_lengths": [len(server.queue) for server in self.servers],
            "server_downlink_queue_lengths": [len(server.downlink_queue) for server in self.servers],
            "cloud_queue_length": len(self.cloud.queue) + len(self.cloud.downlink_queue),
            "active_uploads_by_server": active_uploads_by_server,
            "uplink_data": round(uplink_data, 3),
            "downlink_data": round(downlink_data, 3),
            "network_data": round(
                uplink_data + downlink_data,
                3,
            ),
            "local_compute_used": round(local_compute_used, 3),
            "edge_compute_used": [round(value, 3) for value in edge_compute_used],
            "edge_compute_capacity": [round(value, 3) for value in edge_compute_capacity],
            "edge_utilization": [round(value, 3) for value in edge_utilization],
            "avg_edge_utilization": round(sum(edge_utilization) / max(len(edge_utilization), 1), 3),
            "cloud_compute_used": round(cloud_compute_used, 3),
            "cloud_compute_capacity": round(cloud_compute_capacity, 3),
            "cloud_utilization": round(cloud_utilization, 3),
            "cloud_completed_tasks": cloud_completed_tasks,
            "cloud_usage_ratio": round(cloud_completed_tasks / max(completed, 1), 3),
            "energy_used": round(energy_used, 3),
            "cloud_cost": round(cloud_cost, 3),
            "enable_uplink_contention": self.config.enable_uplink_contention,
            "enable_downlink_transmission": self.config.enable_downlink_transmission,
            "enable_cloud_fallback": self.config.enable_cloud_fallback,
            "completed_by_type": completed_by_type or {},
            "dropped_by_type": dropped_by_type or {},
        }

    def _total_queue_length(self) -> int:
        return (
            sum(len(user.queue) for user in self.users)
            + sum(len(server.queue) for server in self.servers)
            + sum(len(server.downlink_queue) for server in self.servers)
            + len(self.cloud.queue)
            + len(self.cloud.downlink_queue)
        )

    def _count_by_task_type(self, tasks: list[Task]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for task in tasks:
            counts[task.task_type] = counts.get(task.task_type, 0) + 1
        return counts
