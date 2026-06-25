from __future__ import annotations

from .config import MECConfig
from .env import MECEnv
from .gym_wrapper import GymnasiumMECEnv

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - optional dependency
    gym = None

try:
    import gym as legacy_gym
except ImportError:  # pragma: no cover - optional dependency
    legacy_gym = None


def _to_legacy_space(space):
    if legacy_gym is None:
        raise ImportError("_to_legacy_space requires gym to be installed.")
    if gym is not None:
        if isinstance(space, gym.spaces.Box):
            return legacy_gym.spaces.Box(
                low=space.low,
                high=space.high,
                shape=space.shape,
                dtype=space.dtype,
            )
        if isinstance(space, gym.spaces.MultiBinary):
            return legacy_gym.spaces.MultiBinary(space.n)
        if isinstance(space, gym.spaces.Discrete):
            return legacy_gym.spaces.Discrete(space.n)
    return space


class DreamerMECEnv(gym.Env if gym is not None else object):
    """
    Compatibility wrapper for DreamerV3's legacy Gym adapter.

    It exposes the classic Gym API:
    - reset() -> obs
    - step(action) -> obs, reward, done, info
    """

    metadata = {"render_modes": []}

    def __init__(self, action_mode: str = "box", trace_path: str | None = None):
        if gym is None:
            raise ImportError("DreamerMECEnv requires gymnasium to be installed.")
        super().__init__()
        self.env = GymnasiumMECEnv(MECEnv(MECConfig(task_trace_path=trace_path)), action_mode=action_mode)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, *, seed=None, options=None):
        obs, _ = self.env.reset(seed=seed, options=options)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        return obs, reward, done, info

    def render(self, mode="human"):
        del mode
        return self.env.render()

    def close(self):
        return self.env.close()


class LegacyDreamerMECEnv(legacy_gym.Env if legacy_gym is not None else object):
    """Compatibility wrapper for legacy Gym-based callers such as DreamerV3."""

    metadata = {"render.modes": []}

    def __init__(self, action_mode: str = "box", trace_path: str | None = None):
        if legacy_gym is None:
            raise ImportError("LegacyDreamerMECEnv requires gym to be installed.")
        super().__init__()
        self.env = GymnasiumMECEnv(MECEnv(MECConfig(task_trace_path=trace_path)), action_mode=action_mode)
        self.observation_space = _to_legacy_space(self.env.observation_space)
        self.action_space = _to_legacy_space(self.env.action_space)

    def reset(self):
        obs, _ = self.env.reset()
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        return obs, reward, done, info

    def render(self, mode="human"):
        del mode
        return self.env.render()

    def close(self):
        return self.env.close()
