from __future__ import annotations

try:
    from gymnasium.envs.registration import register
    from gymnasium.envs.registration import registry
except ImportError:  # pragma: no cover - optional dependency
    register = None
    registry = None

try:
    from gym.envs.registration import register as legacy_register
    from gym.envs.registration import registry as legacy_registry
except ImportError:  # pragma: no cover - optional dependency
    legacy_register = None
    legacy_registry = None


def register_gym_envs() -> None:
    if register is None:
        raise ImportError("register_gym_envs requires gymnasium to be installed.")

    env_specs = [
        ("MECBox-v0", "box"),
        ("MECMultiBinary-v0", "multibinary"),
    ]

    for env_id, action_mode in env_specs:
        if registry is not None and env_id in registry:
            continue
        register(
            id=env_id,
            entry_point="sim.gym_wrapper:GymnasiumMECEnv",
            kwargs={"action_mode": action_mode},
            disable_env_checker=True,
        )


def register_dreamer_envs() -> None:
    if register is None and legacy_register is None:
        raise ImportError("register_dreamer_envs requires gymnasium or gym to be installed.")

    env_specs = [
        ("MECDreamerBox-v0", "box"),
        ("MECDreamerMultiBinary-v0", "multibinary"),
    ]

    for env_id, action_mode in env_specs:
        if register is not None and (registry is None or env_id not in registry):
            register(
                id=env_id,
                entry_point="sim.dreamer_wrapper:DreamerMECEnv",
                kwargs={"action_mode": action_mode},
                disable_env_checker=True,
            )
        if legacy_register is not None and (legacy_registry is None or env_id not in legacy_registry):
            legacy_register(
                id=env_id,
                entry_point="sim.dreamer_wrapper:LegacyDreamerMECEnv",
                kwargs={"action_mode": action_mode},
            )
