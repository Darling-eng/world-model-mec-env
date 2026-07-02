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


def nearest_edge_policy(obs: dict) -> list[tuple[int, int]]:
    users = sorted(
        obs["users"],
        key=lambda item: (item["queue_length"], item["current_task_remaining_cycles"]),
        reverse=True,
    )
    chosen: list[tuple[int, int]] = []
    for user in users:
        if user["queue_length"] <= 0:
            continue
        server_id = _nearest_reachable_server(obs, user)
        chosen.append((user["user_id"], server_id))
        if len(chosen) >= obs["max_offloads_per_step"]:
            break
    return chosen


def least_loaded_edge_policy(obs: dict) -> list[tuple[int, int]]:
    users = sorted(
        obs["users"],
        key=lambda item: (item["queue_length"], item["current_task_remaining_cycles"]),
        reverse=True,
    )
    chosen: list[tuple[int, int]] = []
    for user in users:
        if user["queue_length"] <= 0:
            continue
        server_id = _least_loaded_reachable_server(obs, user)
        chosen.append((user["user_id"], server_id))
        if len(chosen) >= obs["max_offloads_per_step"]:
            break
    return chosen


def cloud_edge_policy(obs: dict) -> list[tuple[int, int]]:
    users = sorted(
        obs["users"],
        key=lambda item: (item["queue_length"], item["current_task_remaining_cycles"]),
        reverse=True,
    )
    chosen: list[tuple[int, int]] = []
    cloud_enabled = bool((obs.get("cloud") or {}).get("enabled", False))
    for user in users:
        if user["queue_length"] <= 0:
            continue
        server_id = -1 if cloud_enabled else _least_loaded_reachable_server(obs, user)
        chosen.append((user["user_id"], server_id))
        if len(chosen) >= obs["max_offloads_per_step"]:
            break
    return chosen


def _nearest_reachable_server(obs: dict, user: dict) -> int:
    servers = obs.get("servers") or [{"server_id": 0, "position": 0.0, "queue_length": 0}]
    reachable_ids = _reachable_server_ids(user)
    candidates = [server for server in servers if server["server_id"] in reachable_ids] or servers
    best = min(candidates, key=lambda server: (abs(float(user["position"]) - float(server["position"])), server["server_id"]))
    return int(best["server_id"])


def _least_loaded_reachable_server(obs: dict, user: dict) -> int:
    servers = obs.get("servers") or [{"server_id": 0, "position": 0.0, "queue_length": 0}]
    reachable_ids = _reachable_server_ids(user)
    candidates = [server for server in servers if server["server_id"] in reachable_ids] or servers
    best = min(
        candidates,
        key=lambda server: (
            int(server["queue_length"]),
            abs(float(user["position"]) - float(server["position"])),
            int(server["server_id"]),
        ),
    )
    return int(best["server_id"])


def _reachable_server_ids(user: dict) -> set[int]:
    rates = user.get("server_rates") or []
    reachable = {
        int(item["server_id"])
        for item in rates
        if item.get("reachable", True)
    }
    return reachable
