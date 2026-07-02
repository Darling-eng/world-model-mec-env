from dataclasses import dataclass


@dataclass
class MECConfig:
    num_users: int = 10
    max_offloads_per_step: int = 3
    area_size: float = 100.0
    step_duration: float = 1.0
    episode_length: int = 100

    task_arrival_prob: float = 0.35
    task_size_min: float = 4.0
    task_size_max: float = 10.0
    task_cycles_per_unit: float = 6.0
    task_deadline: int = 12
    task_trace_path: str | None = None
    task_type_count: int = 1
    task_type_names: tuple[str, ...] | None = None
    task_type_probabilities: tuple[float, ...] | None = None
    task_cycles_per_unit_by_type: tuple[float, ...] | None = None
    task_deadlines_by_type: tuple[int, ...] | None = None
    task_output_ratios_by_type: tuple[float, ...] | None = None
    task_priorities_by_type: tuple[float, ...] | None = None

    local_compute_rate: float = 4.0
    mec_compute_rate: float = 14.0
    num_edge_servers: int = 1
    edge_server_positions: tuple[float, ...] | None = None
    edge_server_compute_rates: tuple[float, ...] | None = None
    edge_server_coverage_radius: float | None = None
    edge_selection_policy: str = "nearest"
    enable_cloud_fallback: bool = False
    cloud_compute_rate: float = 40.0
    cloud_wan_upload_rate: float = 4.0
    cloud_wan_downlink_rate: float = 6.0
    cloud_wan_delay_steps: int = 2
    local_energy_per_cycle: float = 0.02
    edge_energy_per_cycle: float = 0.01
    cloud_energy_per_cycle: float = 0.008
    network_energy_per_data: float = 0.005
    cloud_cost_per_cycle: float = 0.002

    base_uplink_rate: float = 12.0
    base_downlink_rate: float = 16.0
    pathloss_bias: float = 8.0
    channel_noise: float = 0.15
    enable_uplink_contention: bool = False
    enable_downlink_transmission: bool = False

    reward_preset: str = "debug"
    delay_penalty: float = 1.0
    drop_penalty: float = 4.0
    queue_penalty: float = 0.1
    completion_bonus: float = 1.0
    sla_violation_penalty: float = 6.0

    random_seed: int | None = 7
