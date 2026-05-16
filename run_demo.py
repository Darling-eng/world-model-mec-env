from sim import MECConfig, MECEnv, best_uplink_rate_policy, flatten_observation


def main() -> None:
    env = MECEnv(MECConfig())
    obs, info = env.reset()
    print("Initial info:", info)
    print("Flattened observation dimension:", len(flatten_observation(obs)))

    total_reward = 0.0
    for _ in range(env.config.episode_length):
        action = best_uplink_rate_policy(obs)
        obs, reward, done, step_info = env.step(action)
        total_reward += reward
        print(
            f"step={step_info['step']:03d} "
            f"action={step_info['accepted_action']} "
            f"reward={reward:7.3f} "
            f"completed={step_info['completed_tasks']:2d} "
            f"dropped={step_info['dropped_tasks']:2d} "
            f"queue={step_info['total_queue']:2d}"
        )
        if done:
            break

    print(f"Episode finished. Total reward: {total_reward:.3f}")


if __name__ == "__main__":
    main()
