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

    local_compute_rate: float = 4.0
    mec_compute_rate: float = 14.0

    base_uplink_rate: float = 12.0
    pathloss_bias: float = 8.0
    channel_noise: float = 0.15

    delay_penalty: float = 1.0
    drop_penalty: float = 4.0
    queue_penalty: float = 0.1

    random_seed: int | None = 7
