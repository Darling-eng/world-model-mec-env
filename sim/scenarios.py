from __future__ import annotations

from dataclasses import replace

from .config import MECConfig


SCENARIO_NAMES = (
    "simple_single_edge",
    "multi_edge_sla",
    "multi_edge_heterogeneous_sla",
    "multi_edge_network_sla",
    "cloud_edge_sla",
)


def build_scenario_config(name: str, base: MECConfig | None = None) -> MECConfig:
    config = base or MECConfig()
    if name == "simple_single_edge":
        return replace(
            config,
            reward_preset="sla",
            num_edge_servers=1,
            enable_uplink_contention=False,
            enable_downlink_transmission=False,
        )
    if name == "multi_edge_sla":
        return replace(
            config,
            reward_preset="sla",
            num_edge_servers=3,
            edge_server_compute_rates=(10.0, 14.0, 18.0),
            edge_server_coverage_radius=60.0,
            enable_uplink_contention=False,
            enable_downlink_transmission=False,
        )
    if name == "multi_edge_heterogeneous_sla":
        return replace(
            config,
            reward_preset="sla",
            num_edge_servers=3,
            edge_server_compute_rates=(10.0, 14.0, 18.0),
            edge_server_coverage_radius=60.0,
            task_type_count=3,
            task_type_names=("light", "normal", "urgent"),
            task_type_probabilities=(0.5, 0.3, 0.2),
            task_cycles_per_unit_by_type=(4.0, 6.0, 10.0),
            task_deadlines_by_type=(14, 10, 6),
            task_output_ratios_by_type=(0.05, 0.1, 0.2),
            task_priorities_by_type=(1.0, 2.0, 5.0),
            enable_uplink_contention=False,
            enable_downlink_transmission=False,
        )
    if name == "multi_edge_network_sla":
        return replace(
            build_scenario_config("multi_edge_heterogeneous_sla", config),
            enable_uplink_contention=True,
            enable_downlink_transmission=True,
        )
    if name == "cloud_edge_sla":
        return replace(
            build_scenario_config("multi_edge_network_sla", config),
            enable_cloud_fallback=True,
            cloud_compute_rate=40.0,
            cloud_wan_upload_rate=4.0,
            cloud_wan_downlink_rate=6.0,
            cloud_wan_delay_steps=2,
        )
    raise ValueError(f"Unknown scenario: {name}")
