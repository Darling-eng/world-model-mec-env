from __future__ import annotations

from .env import MECEnv
from .vector import action_to_binary_vector, flatten_observation, observation_vector_length

try:
    import gymnasium as gym
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    gym = None
    np = None


class GymnasiumMECEnv:
    """
    Optional Gymnasium-compatible wrapper.

    The wrapped action space is `MultiBinary(num_users)`. Each active bit means
    "try to offload the head-of-line task for this user in the current step".
    The base environment still applies its own `max_offloads_per_step` cap.
    """

    def __init__(self, env: MECEnv | None = None):
        if gym is None or np is None:
            raise ImportError("GymnasiumMECEnv requires gymnasium and numpy to be installed.")

        self.env = env or MECEnv()
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_vector_length(self.env.config.num_users),),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.MultiBinary(self.env.config.num_users)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del options
        obs, info = self.env.reset(seed=seed)
        return np.asarray(flatten_observation(obs), dtype=np.float32), info

    def step(self, action):
        selected_users = [idx for idx, flag in enumerate(action) if flag]
        obs, reward, done, info = self.env.step(selected_users)
        terminated = False
        truncated = done
        info["binary_action"] = action_to_binary_vector(info["accepted_action"], self.env.config.num_users)
        return (
            np.asarray(flatten_observation(obs), dtype=np.float32),
            float(reward),
            terminated,
            truncated,
            info,
        )

    def render(self):
        return self.env.last_info

    def close(self):
        return None
