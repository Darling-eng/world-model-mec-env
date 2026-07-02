from __future__ import annotations

import os
from pathlib import Path

from .config import MECConfig
from .scenarios import build_scenario_config


def build_runtime_config(
    *,
    scenario: str | None = None,
    trace_path: str | Path | None = None,
    reward_preset: str | None = None,
    seed: int | None = None,
) -> MECConfig:
    config = build_scenario_config(scenario, MECConfig(random_seed=seed)) if scenario else MECConfig(random_seed=seed)
    if trace_path is not None:
        config.task_trace_path = str(trace_path)
    if reward_preset is not None:
        config.reward_preset = reward_preset
    if seed is not None:
        config.random_seed = seed
    return config


def build_runtime_config_from_env(
    *,
    scenario: str | None = None,
    trace_path: str | Path | None = None,
    reward_preset: str | None = None,
    seed: int | None = None,
) -> MECConfig:
    resolved_scenario = scenario if scenario is not None else os.environ.get("MEC_SCENARIO")
    resolved_trace = trace_path if trace_path is not None else os.environ.get("MEC_TRACE_PATH")
    resolved_reward = reward_preset if reward_preset is not None else os.environ.get("MEC_REWARD_PRESET")
    return build_runtime_config(
        scenario=resolved_scenario,
        trace_path=resolved_trace,
        reward_preset=resolved_reward,
        seed=seed,
    )
