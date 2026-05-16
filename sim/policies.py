from __future__ import annotations


def random_policy(obs: dict, rng) -> list[int]:
    max_offloads = int(obs["max_offloads_per_step"])
    user_ids = [user["user_id"] for user in obs["users"]]
    rng.shuffle(user_ids)
    return user_ids[: rng.randint(0, max_offloads)]


def local_only_policy(obs: dict) -> list[int]:
    return []


def best_uplink_rate_policy(obs: dict) -> list[int]:
    users = sorted(obs["users"], key=lambda item: item["uplink_rate"], reverse=True)
    chosen: list[int] = []
    for user in users:
        if user["queue_length"] <= 0:
            continue
        chosen.append(user["user_id"])
        if len(chosen) >= obs["max_offloads_per_step"]:
            break
    return chosen


def largest_queue_policy(obs: dict) -> list[int]:
    users = sorted(
        obs["users"],
        key=lambda item: (item["queue_length"], item["current_task_remaining_cycles"]),
        reverse=True,
    )
    chosen: list[int] = []
    for user in users:
        if user["queue_length"] <= 0:
            continue
        chosen.append(user["user_id"])
        if len(chosen) >= obs["max_offloads_per_step"]:
            break
    return chosen
