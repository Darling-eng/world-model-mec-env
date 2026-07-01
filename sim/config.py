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

    base_uplink_rate: float = 12.0
    pathloss_bias: float = 8.0
    channel_noise: float = 0.15

    reward_preset: str = "debug"
    delay_penalty: float = 1.0
    drop_penalty: float = 4.0
    queue_penalty: float = 0.1
    completion_bonus: float = 1.0
    sla_violation_penalty: float = 6.0

    random_seed: int | None = 7
