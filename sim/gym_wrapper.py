from __future__ import annotations

from .env import MECEnv
from .vector import (
    action_to_binary_vector,
    binary_vector_to_action,
    flatten_observation,
    observation_vector_length,
    score_vector_to_action,
)

try:
    import gymnasium as gym
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    gym = None
    np = None


class GymnasiumMECEnv:
    """
    Optional Gymnasium-compatible wrapper.

    Action modes:
    - `multibinary`: each active bit means "try to offload this user now"
    - `box`: each action entry is a continuous score; the top-k positive users
      are selected, where k=`max_offloads_per_step`

    The base environment still applies its own `max_offloads_per_step` cap.
    """

    def __init__(self, env: MECEnv | None = None, *, action_mode: str = "multibinary"):
        if gym is None or np is None:
            raise ImportError("GymnasiumMECEnv requires gymnasium and numpy to be installed.")

        self.env = env or MECEnv()
        self.action_mode = action_mode
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_vector_length(self.env.config.num_users),),
            dtype=np.float32,
        )
        if action_mode == "multibinary":
            self.action_space = gym.spaces.MultiBinary(self.env.config.num_users)
        elif action_mode == "box":
            self.action_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.env.config.num_users,),
                dtype=np.float32,
            )
        else:
            raise ValueError(f"Unsupported action_mode: {action_mode}")

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del options
        obs, info = self.env.reset(seed=seed)
        return np.asarray(flatten_observation(obs), dtype=np.float32), info

    def step(self, action):
        selected_users = self._decode_action(action)
        obs, reward, done, info = self.env.step(selected_users)
        terminated = False
        truncated = done
        info["raw_action_mode"] = self.action_mode
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

    def _decode_action(self, action) -> list[int]:
        if self.action_mode == "multibinary":
            return binary_vector_to_action(action, self.env.config.num_users)
        if self.action_mode == "box":
            return score_vector_to_action(
                action,
                self.env.config.num_users,
                self.env.config.max_offloads_per_step,
            )
        raise ValueError(f"Unsupported action_mode: {self.action_mode}")
