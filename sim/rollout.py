from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .env import MECEnv
from .vector import flatten_observation


@dataclass
class Transition:
    observation: list[float]
    action: list[int]
    reward: float
    next_observation: list[float]
    done: bool
    info: dict


def collect_episode(
    env: MECEnv,
    policy: Callable[[dict], list[int]],
    *,
    flatten_obs: bool = True,
    seed: int | None = None,
) -> list[Transition]:
    obs, _ = env.reset(seed=seed)
    transitions: list[Transition] = []

    while True:
        action = policy(obs)
        next_obs, reward, done, info = env.step(action)
        current_obs_value = flatten_observation(obs) if flatten_obs else obs
        next_obs_value = flatten_observation(next_obs) if flatten_obs else next_obs
        transitions.append(
            Transition(
                observation=current_obs_value,
                action=list(info["accepted_action"]),
                reward=float(reward),
                next_observation=next_obs_value,
                done=bool(done),
                info=info,
            )
        )
        obs = next_obs
        if done:
            break

    return transitions
