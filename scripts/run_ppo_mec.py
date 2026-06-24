from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim import GymnasiumMECEnv


@dataclass
class PPOConfig:
    total_steps: int = 2_000
    rollout_steps: int = 256
    update_epochs: int = 4
    minibatch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    actor_lr: float = 3e-4
    value_lr: float = 1e-3
    action_std: float = 0.6
    seed: int = 7


class LinearPPOAgent:
    """Small dependency-free PPO actor-critic for MEC smoke experiments."""

    def __init__(self, obs_dim: int, action_dim: int, config: PPOConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.actor_w = self.rng.normal(0.0, 0.01, size=(obs_dim, action_dim))
        self.actor_b = np.zeros(action_dim, dtype=np.float64)
        self.value_w = np.zeros(obs_dim, dtype=np.float64)
        self.value_b = 0.0

    def act(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        obs_batch = normalize_observations(np.asarray(obs, dtype=np.float64)[None, :])
        mean = self._mean(obs_batch)[0]
        raw_action = mean + self.rng.normal(0.0, self.config.action_std, size=mean.shape)
        env_action = np.clip(raw_action, -1.0, 1.0).astype(np.float32)
        logp = float(self._log_prob(raw_action[None, :], mean[None, :])[0])
        value = float(self._value(obs_batch)[0])
        return env_action, raw_action.astype(np.float64), logp, value

    def deterministic_action(self, obs: np.ndarray) -> np.ndarray:
        obs_batch = normalize_observations(np.asarray(obs, dtype=np.float64)[None, :])
        return np.clip(self._mean(obs_batch)[0], -1.0, 1.0).astype(np.float32)

    def update(
        self,
        observations: np.ndarray,
        raw_actions: np.ndarray,
        old_log_probs: np.ndarray,
        returns: np.ndarray,
        advantages: np.ndarray,
    ) -> dict[str, float]:
        x = normalize_observations(observations)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        batch_size = len(observations)
        indices = np.arange(batch_size)
        last_actor_loss = 0.0
        last_value_loss = 0.0

        for _ in range(self.config.update_epochs):
            self.rng.shuffle(indices)
            for start in range(0, batch_size, self.config.minibatch_size):
                batch_idx = indices[start : start + self.config.minibatch_size]
                xb = x[batch_idx]
                action_b = raw_actions[batch_idx]
                old_logp_b = old_log_probs[batch_idx]
                return_b = returns[batch_idx]
                adv_b = advantages[batch_idx]

                z = xb @ self.actor_w + self.actor_b
                mean = np.tanh(z)
                new_logp = self._log_prob(action_b, mean)
                ratio = np.exp(np.clip(new_logp - old_logp_b, -20.0, 20.0))
                clipped_ratio = np.clip(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio)
                surrogate = np.minimum(ratio * adv_b, clipped_ratio * adv_b)
                last_actor_loss = float(-surrogate.mean())

                active = ((adv_b >= 0.0) & (ratio <= 1.0 + self.config.clip_ratio)) | (
                    (adv_b < 0.0) & (ratio >= 1.0 - self.config.clip_ratio)
                )
                coeff = np.where(active, -adv_b * ratio / max(len(batch_idx), 1), 0.0)
                dlogp_dmean = (action_b - mean) / (self.config.action_std**2)
                dz = coeff[:, None] * dlogp_dmean * (1.0 - mean**2)
                grad_actor_w = xb.T @ dz
                grad_actor_b = dz.sum(axis=0)

                values = self._value(xb)
                value_error = values - return_b
                last_value_loss = float(np.mean(value_error**2))
                dvalue = 2.0 * value_error / max(len(batch_idx), 1)
                grad_value_w = xb.T @ dvalue
                grad_value_b = float(dvalue.sum())

                self.actor_w -= self.config.actor_lr * grad_actor_w
                self.actor_b -= self.config.actor_lr * grad_actor_b
                self.value_w -= self.config.value_lr * grad_value_w
                self.value_b -= self.config.value_lr * grad_value_b

        return {
            "actor_loss": last_actor_loss,
            "value_loss": last_value_loss,
            "advantage_mean": float(advantages.mean()),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            actor_w=self.actor_w,
            actor_b=self.actor_b,
            value_w=self.value_w,
            value_b=np.asarray([self.value_b]),
            action_std=np.asarray([self.config.action_std]),
        )

    def _mean(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x @ self.actor_w + self.actor_b)

    def _value(self, x: np.ndarray) -> np.ndarray:
        return x @ self.value_w + self.value_b

    def _log_prob(self, raw_actions: np.ndarray, means: np.ndarray) -> np.ndarray:
        var = self.config.action_std**2
        log_scale = math.log(self.config.action_std)
        return -0.5 * np.sum(
            ((raw_actions - means) ** 2) / var + 2.0 * log_scale + math.log(2.0 * math.pi),
            axis=1,
        )


def normalize_observations(observations: np.ndarray) -> np.ndarray:
    obs = observations.astype(np.float64, copy=True)
    scale = np.ones(obs.shape[1], dtype=np.float64)
    scale[0] = 100.0
    scale[1] = 100.0
    scale[2] = 10.0
    num_users = (obs.shape[1] - 3) // 6
    for user_index in range(num_users):
        offset = 3 + user_index * 6
        scale[offset : offset + 6] = np.asarray([100.0, 10.0, 25.0, 10.0, 80.0, 12.0])
    return obs / scale


def collect_rollout(env: GymnasiumMECEnv, agent: LinearPPOAgent, config: PPOConfig, start_seed: int):
    obs, _ = env.reset(seed=start_seed)
    observations = []
    raw_actions = []
    old_log_probs = []
    values = []
    rewards = []
    dones = []
    episode_returns = []
    episode_completed = []
    episode_dropped = []
    episode_deadline_violations = []
    episode_deadline_rates = []
    episode_delays = []
    episode_queues = []

    current_return = 0.0
    current_completed = 0
    current_dropped = 0
    current_deadline_violations = 0
    current_delay = 0.0
    current_queue = 0.0
    current_steps = 0
    episode_index = 0

    for _ in range(config.rollout_steps):
        env_action, raw_action, logp, value = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(env_action)
        done = bool(terminated or truncated)

        observations.append(obs)
        raw_actions.append(raw_action)
        old_log_probs.append(logp)
        values.append(value)
        rewards.append(float(reward))
        dones.append(done)

        current_return += float(reward)
        current_completed += int(info["completed_tasks"])
        current_dropped += int(info["dropped_tasks"])
        current_deadline_violations += int(info.get("deadline_violations", info["dropped_tasks"]))
        current_delay += float(info["avg_delay"])
        current_queue += float(info["total_queue"])
        current_steps += 1

        obs = next_obs
        if done:
            episode_returns.append(current_return)
            episode_completed.append(current_completed)
            episode_dropped.append(current_dropped)
            episode_deadline_violations.append(current_deadline_violations)
            episode_deadline_rates.append(
                current_deadline_violations / max(current_completed + current_deadline_violations, 1)
            )
            episode_delays.append(current_delay / max(current_steps, 1))
            episode_queues.append(current_queue / max(current_steps, 1))

            episode_index += 1
            obs, _ = env.reset(seed=start_seed + episode_index)
            current_return = 0.0
            current_completed = 0
            current_dropped = 0
            current_deadline_violations = 0
            current_delay = 0.0
            current_queue = 0.0
            current_steps = 0

    last_value = 0.0 if dones[-1] else float(agent._value(normalize_observations(obs[None, :]))[0])
    advantages, returns = compute_gae(
        np.asarray(rewards, dtype=np.float64),
        np.asarray(values, dtype=np.float64),
        np.asarray(dones, dtype=bool),
        last_value,
        config,
    )

    metrics = {
        "rollout_reward": float(np.mean(episode_returns)) if episode_returns else current_return,
        "completed_tasks": float(np.mean(episode_completed)) if episode_completed else float(current_completed),
        "dropped_tasks": float(np.mean(episode_dropped)) if episode_dropped else float(current_dropped),
        "deadline_violations": float(np.mean(episode_deadline_violations))
        if episode_deadline_violations
        else float(current_deadline_violations),
        "deadline_violation_rate": float(np.mean(episode_deadline_rates))
        if episode_deadline_rates
        else current_deadline_violations / max(current_completed + current_deadline_violations, 1),
        "avg_delay": float(np.mean(episode_delays)) if episode_delays else current_delay / max(current_steps, 1),
        "avg_total_queue": float(np.mean(episode_queues)) if episode_queues else current_queue / max(current_steps, 1),
    }

    return (
        np.asarray(observations, dtype=np.float64),
        np.asarray(raw_actions, dtype=np.float64),
        np.asarray(old_log_probs, dtype=np.float64),
        returns,
        advantages,
        metrics,
    )


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    config: PPOConfig,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float64)
    gae = 0.0
    for step in reversed(range(len(rewards))):
        next_value = last_value if step == len(rewards) - 1 else values[step + 1]
        next_nonterminal = 1.0 - float(dones[step])
        delta = rewards[step] + config.gamma * next_value * next_nonterminal - values[step]
        gae = delta + config.gamma * config.gae_lambda * next_nonterminal * gae
        advantages[step] = gae
    returns = advantages + values
    return advantages, returns


def evaluate(agent: LinearPPOAgent, episodes: int, seed: int) -> dict[str, float]:
    env = GymnasiumMECEnv(action_mode="box")
    results = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        completed = 0
        dropped = 0
        deadline_violations = 0
        total_delay = 0.0
        total_queue = 0.0
        steps = 0
        while True:
            action = agent.deterministic_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            completed += int(info["completed_tasks"])
            dropped += int(info["dropped_tasks"])
            deadline_violations += int(info.get("deadline_violations", info["dropped_tasks"]))
            total_delay += float(info["avg_delay"])
            total_queue += float(info["total_queue"])
            steps += 1
            if terminated or truncated:
                break
        results.append(
            {
                "total_reward": total_reward,
                "completed_tasks": completed,
                "dropped_tasks": dropped,
                "deadline_violations": deadline_violations,
                "deadline_violation_rate": deadline_violations / max(completed + deadline_violations, 1),
                "avg_delay": total_delay / max(steps, 1),
                "avg_total_queue": total_queue / max(steps, 1),
            }
        )
    env.close()
    return {
        key: float(np.mean([item[key] for item in results]))
        for key in [
            "total_reward",
            "completed_tasks",
            "dropped_tasks",
            "deadline_violations",
            "deadline_violation_rate",
            "avg_delay",
            "avg_total_queue",
        ]
    }


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight PPO baseline on the MEC environment.")
    parser.add_argument("--steps", type=int, default=2_000, help="Total environment steps.")
    parser.add_argument("--rollout-steps", type=int, default=256, help="Steps collected before each PPO update.")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Evaluation episodes after training.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/ppo_mec"), help="Directory for logs and model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PPOConfig(
        total_steps=args.steps,
        rollout_steps=args.rollout_steps,
        seed=args.seed,
    )
    env = GymnasiumMECEnv(action_mode="box")
    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])
    agent = LinearPPOAgent(obs_dim, action_dim, config)

    metrics_path = args.log_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    env_steps = 0
    update_index = 0
    while env_steps < config.total_steps:
        rollout_seed = config.seed + update_index * 10_000
        observations, raw_actions, old_log_probs, returns, advantages, rollout_metrics = collect_rollout(
            env, agent, config, rollout_seed
        )
        update_metrics = agent.update(observations, raw_actions, old_log_probs, returns, advantages)
        env_steps += len(observations)
        update_index += 1
        row = {
            "update": update_index,
            "env_steps": env_steps,
            **rollout_metrics,
            **update_metrics,
        }
        append_jsonl(metrics_path, row)
        print(
            f"update={update_index} steps={env_steps} reward={row['rollout_reward']:.3f} "
            f"completed={row['completed_tasks']:.2f} dropped={row['dropped_tasks']:.2f} "
            f"ddl_rate={row['deadline_violation_rate']:.3f} "
            f"avg_delay={row['avg_delay']:.3f} avg_total_queue={row['avg_total_queue']:.3f}"
        )

    eval_metrics = evaluate(agent, args.eval_episodes, seed=config.seed + 1_000_000)
    eval_row = {"phase": "eval", "episodes": args.eval_episodes, **eval_metrics}
    append_jsonl(metrics_path, eval_row)
    agent.save(args.log_dir / "ppo_linear_model.npz")
    env.close()

    print(
        f"eval episodes={args.eval_episodes} reward={eval_metrics['total_reward']:.3f} "
        f"completed={eval_metrics['completed_tasks']:.2f} dropped={eval_metrics['dropped_tasks']:.2f} "
        f"ddl_rate={eval_metrics['deadline_violation_rate']:.3f} "
        f"avg_delay={eval_metrics['avg_delay']:.3f} avg_total_queue={eval_metrics['avg_total_queue']:.3f}"
    )
    print(f"wrote_metrics={metrics_path}")


if __name__ == "__main__":
    main()
