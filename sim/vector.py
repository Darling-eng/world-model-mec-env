from __future__ import annotations

from typing import Iterable


def observation_vector_length(num_users: int) -> int:
    per_user_features = 6
    global_features = 3
    return global_features + num_users * per_user_features


def flatten_observation(obs: dict) -> list[float]:
    vector: list[float] = [
        float(obs["step"]),
        float(obs["server_queue_length"]),
        float(obs["max_offloads_per_step"]),
    ]
    for user in obs["users"]:
        vector.extend(
            [
                float(user["position"]),
                float(user["velocity"]),
                float(user["queue_length"]),
                float(user["current_task_size"]),
                float(user["current_task_remaining_cycles"]),
                float(user["uplink_rate"]),
            ]
        )
    return vector


def action_to_binary_vector(action: Iterable[int], num_users: int) -> list[int]:
    vector = [0] * num_users
    for user_id in action:
        if 0 <= user_id < num_users:
            vector[user_id] = 1
    return vector
