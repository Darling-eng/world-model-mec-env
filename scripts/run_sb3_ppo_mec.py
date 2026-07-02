from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim import GymnasiumMECEnv, MECEnv, SCENARIO_NAMES, build_runtime_config


def make_env(
    trace_path: Path | None = None,
    reward_preset: str = "debug",
    seed: int = 7,
    scenario: str | None = None,
) -> GymnasiumMECEnv:
    config = build_runtime_config(
        scenario=scenario,
        trace_path=trace_path,
        reward_preset=reward_preset,
        seed=seed,
    )
    return GymnasiumMECEnv(MECEnv(config), action_mode="box")


def evaluate(model, episodes: int, seed: int, trace_path: Path | None, reward_preset: str, scenario: str | None) -> dict[str, float]:
    env = make_env(trace_path, reward_preset, seed, scenario)
    rows = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        completed = 0
        dropped = 0
        deadline_violations = 0
        total_delay = 0.0
        total_queue = 0.0
        total_edge_utilization = 0.0
        total_network_data = 0.0
        total_cloud_utilization = 0.0
        total_cloud_usage_ratio = 0.0
        total_energy_used = 0.0
        total_cloud_cost = 0.0
        steps = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            completed += int(info["completed_tasks"])
            dropped += int(info["dropped_tasks"])
            deadline_violations += int(info.get("deadline_violations", info["dropped_tasks"]))
            total_delay += float(info["avg_delay"])
            total_queue += float(info["total_queue"])
            total_edge_utilization += float(info.get("avg_edge_utilization", 0.0))
            total_network_data += float(info.get("network_data", 0.0))
            total_cloud_utilization += float(info.get("cloud_utilization", 0.0))
            total_cloud_usage_ratio += float(info.get("cloud_usage_ratio", 0.0))
            total_energy_used += float(info.get("energy_used", 0.0))
            total_cloud_cost += float(info.get("cloud_cost", 0.0))
            steps += 1
            if terminated or truncated:
                break

        rows.append(
            {
                "total_reward": total_reward,
                "completed_tasks": completed,
                "dropped_tasks": dropped,
                "deadline_violations": deadline_violations,
                "deadline_violation_rate": deadline_violations / max(completed + deadline_violations, 1),
                "avg_delay": total_delay / max(steps, 1),
                "avg_total_queue": total_queue / max(steps, 1),
                "avg_edge_utilization": total_edge_utilization / max(steps, 1),
                "avg_network_data": total_network_data / max(steps, 1),
                "avg_cloud_utilization": total_cloud_utilization / max(steps, 1),
                "avg_cloud_usage_ratio": total_cloud_usage_ratio / max(steps, 1),
                "avg_energy_used": total_energy_used / max(steps, 1),
                "avg_cloud_cost": total_cloud_cost / max(steps, 1),
            }
        )

    env.close()
    keys = [
        "total_reward",
        "completed_tasks",
        "dropped_tasks",
        "deadline_violations",
        "deadline_violation_rate",
        "avg_delay",
        "avg_total_queue",
        "avg_edge_utilization",
        "avg_network_data",
        "avg_cloud_utilization",
        "avg_cloud_usage_ratio",
        "avg_energy_used",
        "avg_cloud_cost",
    ]
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Stable-Baselines3 PPO baseline on the MEC environment.")
    parser.add_argument("--steps", type=int, default=50_000, help="Total PPO training timesteps.")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Evaluation episodes after training.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("experiment_records/sb3_ppo_mec"),
        help="Directory for logs and model.",
    )
    parser.add_argument("--trace", type=Path, default=None, help="Optional normalized task trace CSV.")
    parser.add_argument("--scenario", choices=SCENARIO_NAMES, default=None, help="Optional named MEC scenario.")
    parser.add_argument(
        "--reward-preset",
        choices=["debug", "sla"],
        default="debug",
        help="Reward preset. debug preserves old experiments; sla emphasizes completion and deadline safety.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="PPO learning rate.")
    parser.add_argument("--n-steps", type=int, default=1024, help="PPO rollout steps before each update.")
    parser.add_argument("--batch-size", type=int, default=256, help="PPO minibatch size.")
    parser.add_argument("--n-epochs", type=int, default=10, help="PPO optimization epochs per rollout.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda.")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range.")
    parser.add_argument("--ent-coef", type=float, default=0.0, help="Entropy coefficient.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:  # pragma: no cover - optional Colab dependency
        raise ImportError(
            "run_sb3_ppo_mec.py requires stable-baselines3. Install with: pip install stable-baselines3"
        ) from exc

    args.log_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.log_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    env = Monitor(make_env(args.trace, args.reward_preset, args.seed, args.scenario))
    model = PPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        verbose=1,
    )
    model.learn(total_timesteps=args.steps, progress_bar=False)
    model.save(args.log_dir / "ppo_model")
    env.close()

    eval_metrics = evaluate(
        model,
        episodes=args.eval_episodes,
        seed=args.seed + 1_000_000,
        trace_path=args.trace,
        reward_preset=args.reward_preset,
        scenario=args.scenario,
    )
    row = {
        "algorithm": "sb3_ppo",
        "phase": "eval",
        "episodes": args.eval_episodes,
        "seed": args.seed,
        "scenario": args.scenario or "custom",
        **eval_metrics,
    }
    append_jsonl(metrics_path, row)
    print(
        f"eval episodes={args.eval_episodes} reward={eval_metrics['total_reward']:.3f} "
        f"completed={eval_metrics['completed_tasks']:.2f} dropped={eval_metrics['dropped_tasks']:.2f} "
        f"ddl_rate={eval_metrics['deadline_violation_rate']:.3f} "
        f"avg_delay={eval_metrics['avg_delay']:.3f} avg_total_queue={eval_metrics['avg_total_queue']:.3f}"
    )
    print(f"wrote_metrics={metrics_path}")


if __name__ == "__main__":
    main()
