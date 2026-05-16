from __future__ import annotations

from .gym_wrapper import GymnasiumMECEnv

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - optional dependency
    gym = None

try:
    import gym as legacy_gym
except ImportError:  # pragma: no cover - optional dependency
    legacy_gym = None


class DreamerMECEnv(gym.Env if gym is not None else object):
    """
    Compatibility wrapper for DreamerV3's legacy Gym adapter.

    It exposes the classic Gym API:
    - reset() -> obs
    - step(action) -> obs, reward, done, info
    """

    metadata = {"render_modes": []}

    def __init__(self, action_mode: str = "box"):
        if gym is None:
            raise ImportError("DreamerMECEnv requires gymnasium to be installed.")
        super().__init__()
        self.env = GymnasiumMECEnv(action_mode=action_mode)
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

    def __init__(self, action_mode: str = "box"):
        if legacy_gym is None:
            raise ImportError("LegacyDreamerMECEnv requires gym to be installed.")
        super().__init__()
        self.env = GymnasiumMECEnv(action_mode=action_mode)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

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
