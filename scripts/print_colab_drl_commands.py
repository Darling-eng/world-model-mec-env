from __future__ import annotations

import argparse


SCENARIOS = ("multi_edge_network_sla", "cloud_edge_sla")
SEEDS = (7, 17, 27, 37, 47)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print Colab DRL batch commands for MEC experiments.")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--trace", default="experiment_records/trace_profiles/alibaba_light_normal_urgent.csv")
    parser.add_argument("--dreamer-dir", default="/content/dreamerv3")
    return parser.parse_args()


def print_sb3_ppo(args: argparse.Namespace) -> None:
    print("# Stable-Baselines3 PPO")
    for scenario in SCENARIOS:
        for seed in SEEDS:
            print(
                "python scripts/run_sb3_ppo_mec.py "
                f"--steps {args.steps} "
                f"--eval-episodes {args.eval_episodes} "
                f"--seed {seed} "
                f"--scenario {scenario} "
                f"--trace {args.trace} "
                "--reward-preset sla "
                f"--log-dir experiment_records/sb3_ppo_mec/{scenario}_seed{seed}_{args.steps // 1000}k"
            )


def print_sac(args: argparse.Namespace) -> None:
    print("\n# SAC")
    for scenario in SCENARIOS:
        for seed in SEEDS:
            print(
                "python scripts/run_sac_mec.py "
                f"--steps {args.steps} "
                f"--eval-episodes {args.eval_episodes} "
                f"--seed {seed} "
                f"--scenario {scenario} "
                f"--trace {args.trace} "
                "--reward-preset sla "
                f"--log-dir experiment_records/sac_mec/{scenario}_seed{seed}_{args.steps // 1000}k"
            )


def print_dreamer(args: argparse.Namespace) -> None:
    print("\n# DreamerV3")
    for scenario in SCENARIOS:
        for seed in SEEDS:
            print(
                "python scripts/run_dreamer_mec.py "
                f"--dreamer-dir {args.dreamer_dir} "
                f"--scenario {scenario} "
                f"--trace {args.trace} "
                "--reward-preset sla "
                "--configs debug "
                "--task gym_MECDreamerBox-v0 "
                "--run.envs 1 "
                "--run.eval_envs 0 "
                f"--run.steps {args.steps} "
                f"--seed {seed}"
            )


def main() -> None:
    args = parse_args()
    print_sb3_ppo(args)
    print_sac(args)
    print_dreamer(args)


if __name__ == "__main__":
    main()
